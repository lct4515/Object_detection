import argparse
import os.path
from tkinter import *
from tkinter import filedialog
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from PIL import ImageTk, Image
import cv2
import time

FONT = "黑体"
IMAGE_SIZE = (350, 242)
app = Tk()


def file_select():
    global photo1, img_path
    img_path = filedialog.askopenfilename()
    photo1 = ImageTk.PhotoImage(Image.open(img_path).resize(IMAGE_SIZE))
    origin_photo.configure(image=photo1)


class LoadImagesSingle:  # for inference
    def __init__(self, path, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [img_path]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


def img_recognition():
    start_time = time.time()
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImagesSingle(img_path, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1]:
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':

                    global save_path_temp
                    save_path_temp = save_path
                    cv2.imwrite(save_path, im0)


                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    end_time = time.time()
    img = Image.open(save_path_temp)
    # img.show()
    global photo2, detect_status, detect_time
    s = img.size
    re_s = img.resize(IMAGE_SIZE)
    photo2 = ImageTk.PhotoImage(re_s)
    detect_photo.configure(image=photo2)
    detect_status.configure(text="检测状态： 成功")
    detect_time.configure(text=f"检测状态： {round(end_time - start_time, 5)} 秒")

    app.mainloop()
    # if save_txt or save_img:
    #     #     print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     #     if platform == 'darwin':  # MacOS
    #     #         os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


def video_detect():
    app.quit()
    video_detect_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ui-video.py")
    print(video_detect_path)
    os.system(f"python {video_detect_path}")


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='weights path')
parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='output1', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()
opt.cfg = check_file(opt.cfg)  # check file
opt.names = check_file(opt.names)  # check file
print(opt)

app.title('图片检测')
app.geometry('1000x600')
app.resizable(width=False, height=False)
photo = ImageTk.PhotoImage(Image.open('background.jpg').resize((1000, 600)))
Label(image=photo).place(x=0, y=0)

# 第一行标题
Label(app, text="国能智深目标检测系统", font=(FONT, 30), height=2).pack()

# 第一块检测选择图片的选项
people_button = Button(app, text="检测行人", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 0)
car_button = Button(app, text="检测汽车", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 1)
toothbrush_button = Button(app, text="检测椅子", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 2)
banana_button = Button(app, text="检测盆栽", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 3)
phone_button = Button(app, text="检测屏幕", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 4)
book_button = Button(app, text="检测汤匙", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 5)
comprehensive_button = Button(app, text="综合检测", font=(FONT, 20), command=file_select).place(x=50, y=100 + 60 * 6)

# 第二块原图
photo1 = ImageTk.PhotoImage(Image.open('defalut.jpg').resize(IMAGE_SIZE))
origin_photo = Label(image=photo1)
origin_photo.place(x=200, y=100)
Label(app, text="原图", font=(FONT, 20)).place(
    x=200 + IMAGE_SIZE[0] / 2 - 20,
    y=100 + IMAGE_SIZE[1] + 8
)
detect_status = Label(app, text="检测状态： 无", font=(FONT, 25))
detect_status.place(
    x=200 + 50,
    y=100 + IMAGE_SIZE[1] + 60
)
detect_time = Label(app, text="检测用时： 无", font=(FONT, 25))
detect_time.place(
    x=200 + 50,
    y=100 + IMAGE_SIZE[1] + 60 + 50
)

# 第三块检测
photo2 = ImageTk.PhotoImage(Image.open('defalut.jpg').resize(IMAGE_SIZE))
detect_photo = Label(image=photo2)
detect_photo.place(x=200 + IMAGE_SIZE[0] + 60, y=100)
detect_button = Button(
    text='检测',
    font=(FONT, 40),
    command=img_recognition,
    width=10
).place(
    x=200 + IMAGE_SIZE[0] + 100,
    y=100 + IMAGE_SIZE[1] + 50
)
# 视频检测跳转
video_button = Button(
    app,
    text="视频检测",
    font=(FONT, 20),
    command=video_detect,
    width=8
).place(x=850, y=550)

# 主事件循环
mainloop()
