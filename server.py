
import http.server
import cgi
import os
import re  
import shutil
import cv2
import stat
from jinja2 import Template
import numpy as np
import torch
from collections import Counter
import zipfile
from ultralytics import YOLO






# ディレクトリ設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'movie')
SNAPS_DIR = os.path.join(BASE_DIR, 'snaps')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputimage')
TEMPLATE_HTML = os.path.join(BASE_DIR, 'index.html')
MOVIE_DIR = os.path.join(BASE_DIR, "movie")



# フォルダ初期化
def natural_key(filename):
    # 例: img12.png → ['img', 12, '.png']
        return [
        int(text) if text.isdigit() else text
        for text in re.split(r'(\d+)', filename)
        ]

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=handle_remove_readonly)
    os.makedirs(path)
def extract_blackboard(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])  # 黒板の緑色範囲（調整可能）
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:  # 十分大きな黒板領域か確認
        return None

    x, y, w, h = cv2.boundingRect(largest)
    print(f"largest: {largest.shape}")
    print(cv2.boundingRect)
    blackboard = frame[y:y+h, x:x+w]
    return blackboard

# --- 動画からフレーム抽出、黒板のみ保存 ---
def movie2image(filename):
    os.makedirs(SNAPS_DIR, exist_ok=True)
    movie_path = os.path.join(MOVIE_DIR, filename)
    if not os.path.isfile(movie_path):
        print(f"動画ファイルがありません: {movie_path}")
        return False

    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        print(f"動画を開けません: {movie_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1# 1秒ごとにフレーム抽出
    print(frame_interval)
    

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            blackboard = extract_blackboard(frame)
            if blackboard is not None:
                filename_out = f"blackboard_{count:05d}.jpg"
                save_path = os.path.join(SNAPS_DIR, filename_out)
                if cv2.imwrite(save_path, blackboard):
                    saved += 1
        count += 1
    cap.release()
    if saved == 0:
        print("黒板画像を保存できませんでした。")
        return False
    print(f"保存した黒板画像数: {saved}")
    return True
# --- YOLOで人物領域透明化＋アルファ合成＋最頻色検出＋保存 ---
def merge_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = sorted([f for f in os.listdir(SNAPS_DIR) if f.lower().endswith(('.jpg', '.png'))])
    if not image_files:
        print("画像が見つかりません。")
        return

    standard_size = None
    images = []
    for f in image_files:
        path = os.path.join(SNAPS_DIR, f)
        img = cv2.imread(path)
        if img is None:
            print(f"画像読み込み失敗: {path}")
            continue
        if standard_size is None:
            standard_size = (img.shape[1], img.shape[0])
        else:
            img = cv2.resize(img, standard_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        images.append(img)

    if not images:
        print("有効な画像がありません。")
        return

    print(f"{len(images)} 枚の画像を読み込みました。")

    # YOLOv8モデル読み込み
    model = YOLO(r"C:\Users\harut\OneDrive\Desktop\AI\opencv\models\yolov8n.pt")
    print("YOLOv8モデル読み込み完了")

    # 人物検出して透明化
    for i, img in enumerate(images):
        results = model(img)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        persons = [boxes[j] for j in range(len(boxes)) if int(classes[j]) == 0]

        for p in persons:
            x1, y1, x2, y2 = map(int, p[:4])
            img[y1:y2, x1:x2] = (255, 255, 255, 0)

        images[i] = img

    print("人物透明化完了")

    # 前の画像の透明領域に前のフレームを合成
    for i in range(len(images) - 1):
        if images[i+1].shape != images[i].shape:
            print(f"サイズ不一致: images[{i}]: {images[i].shape}, images[{i+1}]: {images[i+1].shape}")
            continue  # スキップするか、resizeしても良い
        mask = images[i+1][:, :, 3] == 0
        images[i+1][mask] = images[i][mask]

    # 最頻値色検出
    def get_pixel_count(img):
        return img.shape[0] * img.shape[1]
#　最頻値の定義
    for idx, img in enumerate(images):
        output_path = os.path.join(OUTPUT_DIR, f"output_{idx}.png")

        # Hue（色相）の範囲指定

# BGR → HSV に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_hue = 40  # 下限（例：緑色あたり）
        upper_hue = 140  # 上限
        lower_bound = (lower_hue, 50, 30)
        upper_bound = (upper_hue, 150, 80)
        
        # 指定範囲内のマスク作成
        limit = cv2.inRange(hsv, lower_bound, upper_bound)
        pixels = get_pixel_count(hsv)
        print(cv2.countNonZero(limit) > (pixels * 0.3))
        # 条件に合うピクセルが存在するかチェック
        if cv2.countNonZero(limit) <pixels*0.3:
            # 条件を満たす画像だけ保存（元画像そのまま）
            bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imwrite(output_path, bgr)
            print(f"保存: {output_path}")


def zip_dir(input_dir, output_zip):
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                full_path = os.path.join(root, file)
                relative = os.path.relpath(full_path, input_dir)
                zf.write(full_path, relative)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.render_index().encode('utf-8'))
        elif self.path.startswith('/outputimage/'):
            # OUTPUT_DIR 内のファイルを返す
            filepath = os.path.join(OUTPUT_DIR, self.path[len('/outputimage/'):])
            if os.path.exists(filepath):
                self.send_response(200)
                # 拡張子によって Content-Type を分ける
                if filepath.lower().endswith('.mp4'):
                    self.send_header('Content-Type', 'video/mp4')
                else:
                    self.send_header('Content-Type', 'image/png')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_error(404, 'File not found')
        else:
            super().do_GET()
    def do_POST(self):
        if self.path == '/upload':
            clear_folder(UPLOAD_DIR)
            clear_folder(SNAPS_DIR)
            clear_folder(IMAGES_DIR)
            clear_folder(OUTPUT_DIR)

            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                    environ={'REQUEST_METHOD': 'POST',
                                             'CONTENT_TYPE': self.headers['Content-Type']})
            files = form["file"]
            if not isinstance(files, list):
                files = [files]

            for file in files:
                if file.filename:
                    filename = os.path.basename(file.filename)
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    with open(filepath, 'wb') as f:
                        shutil.copyfileobj(file.file, f)
                    print(f"保存: {filepath}")
                    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.png')):
                        movie2image(filename)
                else:
                    print("無効なファイル")

            for img in os.listdir(SNAPS_DIR):
                src = os.path.join(SNAPS_DIR, img)
                dst = os.path.join(IMAGES_DIR, img)
                shutil.copy(src, dst)

            merge_images()

            zip_dir("outputimage", "outputimage.zip")

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.render2_index().encode('utf-8'))
        else:
            self.send_error(404)
    


# ファイル名のリスト（例）

    def render_index(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        files=[]
        # HTMLギャラリー生成
        gallery = ''.join([
            f'<div>'
            f'<img src="/outputimage/{f}" width="200">' if not f.lower().endswith('.mp4')
            else f'<video src="/outputimage/{f}" width="200" controls></video>'
            f'</div>'
            for f in files
        ])

        # テンプレート読み込み（Jinja2で処理）
        with open(TEMPLATE_HTML, encoding='utf-8') as f:
            template = Template(f.read())

        # {{ gallery|safe }} がちゃんとHTML展開される
        return template.render(gallery=gallery)
    
    def render2_index(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 出力ディレクトリ内の対象ファイル取得
        files = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith(('.jpg', '.png', '.mp4'))
        ]

        # 自然順ソート（natsorted 不使用）
        files = sorted(files, key=natural_key)

        print("ファイル一覧:", files)

        # HTMLギャラリー生成
        gallery = ''.join([
            (
                f'<div>'
                f'<img src="/outputimage/{f}" width="200">'
                f'</div>'
                if not f.lower().endswith('.mp4')
                else
                f'<div>'
                f'<video src="/outputimage/{f}" width="200" controls></video>'
                f'</div>'
            )
            for f in files
        ])
        # テンプレート読み込み（Jinja2で処理）
        with open(TEMPLATE_HTML, encoding='utf-8') as f:
            template = Template(f.read())

        # {{ gallery|safe }} がちゃんとHTML展開される
        return template.render(gallery=gallery)
    


if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    server = http.server.HTTPServer(('', 8000), Handler)
    print("サーバ起動: http://localhost:8000")
    server.serve_forever()
