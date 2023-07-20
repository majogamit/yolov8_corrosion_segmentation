from ultralytics import YOLO

model = YOLO('best_so_far.pt')

model.predict(source=['0.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg'], show=True, save=True, show_labels=True, conf=0.5, save_txt=False, save_crop=False, line_width=2, box=True, visualize=False)
