import pyautogui
import cv2
import numpy as np
import onnxruntime as ort
import time
from pynput import keyboard

# ================= НАСТРОЙКИ =================
MODEL_PATH = 'bestV1.onnx'
CONF_THRESHOLD = 0.15       
IOU_THRESHOLD = 0.4
COOKIE_CLICKS = 10
CLICK_DELAY = 0.05
IMG_SIZE = 640

# Задержки, чтобы не кликать по одному и тому же объекту бесконечно
UPGRADE_COOLDOWN = 2.0  # Секунд после клика по улучшению
RECLAMA_COOLDOWN = 1.0  # Секунд после клика по рекламе
# =============================================

stop_bot = False
bot_running = False
last_upgrade_click = 0
last_reclama_click = 0


def on_press(key):
    global stop_bot, bot_running
    try:
        if key == keyboard.Key.space:
            if bot_running:
                print("\n[!] ПРОБЕЛ — останавливаю бота...")
                stop_bot = True
            else:
                print("\n[!] ПРОБЕЛ — запускаю бота...")
                stop_bot = False
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press)
listener.start()

pyautogui.FAILSAFE = True

print("Загрузка ONNX модели...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASS_NAMES = ['game', 'upgrade', 'not reclama']


def preprocess(image, img_size=IMG_SIZE):
    img = cv2.resize(image, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img


def postprocess(outputs, conf_threshold, orig_w, orig_h, img_size=IMG_SIZE):
    predictions = np.squeeze(outputs[0]).T
    found = {name: None for name in CLASS_NAMES}
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        classes_scores = pred[4:]
        class_id = int(np.argmax(classes_scores))
        max_score = float(classes_scores[class_id])

        if max_score <= conf_threshold:
            continue

        x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]

        x1 = int((x_center - w / 2) * orig_w / img_size)
        y1 = int((y_center - h / 2) * orig_h / img_size)
        x2 = int((x_center + w / 2) * orig_w / img_size)
        y2 = int((y_center + h / 2) * orig_h / img_size)

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(max_score)
        class_ids.append(class_id)

    if not boxes:
        return found

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, IOU_THRESHOLD)
    if len(indices) == 0:
        return found

    indices = np.array(indices).flatten()

    for i in indices:
        x, y, w, h = boxes[i]
        cls_name = CLASS_NAMES[class_ids[i]]
        conf = scores[i]
        cx, cy = x + w / 2, y + h / 2

        if found[cls_name] is None or conf > found[cls_name]['conf']:
            found[cls_name] = {'conf': conf, 'x': cx, 'y': cy}

    return found


def click(x, y):
    pyautogui.moveTo(int(x), int(y), duration=0)
    pyautogui.click()


def find_and_click():
    global stop_bot, last_upgrade_click, last_reclama_click
    if stop_bot:
        return False

    now = time.time()

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    orig_h, orig_w = screenshot_np.shape[:2]
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    input_tensor = preprocess(screenshot_bgr)
    outputs = session.run([output_name], {input_name: input_tensor})
    found = postprocess(outputs, CONF_THRESHOLD, orig_w, orig_h)

    # Вывод в консоль (для отладки)
    for cls, data in found.items():
        if data:
            print(f"[Вижу] {cls:12s} | conf={data['conf']:.2f} | "
                  f"X={data['x']:.0f} Y={data['y']:.0f}")

    # --- 1. ЗАКРЫВАЕМ РЕКЛАМУ (если она есть и не было клика недавно) ---
    if found['not reclama'] and (now - last_reclama_click > RECLAMA_COOLDOWN):
        click(found['not reclama']['x'], found['not reclama']['y'])
        print(">>> Клик по РЕКЛАМЕ")
        last_reclama_click = now
        time.sleep(0.5)
        return True

    # --- 2. ПОКУПАЕМ УЛУЧШЕНИЕ (если есть и не было клика недавно) ---
    if found['upgrade'] and (now - last_upgrade_click > UPGRADE_COOLDOWN):
        click(found['upgrade']['x'], found['upgrade']['y'])
        print(">>> Клик по УЛУЧШЕНИЮ")
        last_upgrade_click = now
        time.sleep(0.5)
        return True

    # --- 3. КЛИКАЕМ ПО ПЕЧЕНЬКЕ (всегда, если она есть) ---
    if found['game']:
        x, y = found['game']['x'], found['game']['y']
        pyautogui.moveTo(int(x), int(y), duration=0)
        for _ in range(COOKIE_CLICKS):
            if stop_bot:
                return False
            pyautogui.click()
            time.sleep(CLICK_DELAY)
        print(f">>> {COOKIE_CLICKS} кликов по ПЕЧЕНЬКЕ")
        return True

    return False


def bot_loop():
    while not stop_bot:
        try:
            if not find_and_click():
                time.sleep(0.3)
        except pyautogui.FailSafeException:
            print("\nОстановлено (мышь в углу).")
            return
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(1)


def main():
    global stop_bot, bot_running

    print("==========================================")
    print("ПРОБЕЛ — старт / стоп.  Ctrl+C — выход.")
    print("==========================================")
    print("Бот в режиме ожидания. Нажмите ПРОБЕЛ для запуска.")

    try:
        while True:
            if not stop_bot and not bot_running:
                bot_running = True
                print("\n>>> Бот ЗАПУЩЕН")
                bot_loop()
                bot_running = False
                print("\n>>> Бот ОСТАНОВЛЕН. ПРОБЕЛ — запустить снова.")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nВыход (Ctrl+C).")
    finally:
        listener.stop()
        print("Программа завершена.")


if __name__ == "__main__":
    main()