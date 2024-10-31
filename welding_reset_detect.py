import cv2
import torch
from shapely.geometry import box, Polygon

from datetime import datetime
from ultralytics import YOLO

from config import VIDEOS_WELDING,WEIGHTS_WELDING_RESET,SAVE_IMG_PATH_WELDING_K2,URL_IMG_PATH_WELDING_K2,WELDING_REGION
import logging
from uvicorn.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logging = logging.getLogger("uvicorn")
def video_decoder(rtsp_url, frame_queue_list,start_event, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            logging.info("Video streaming is closing❎")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
            continue
        if rtsp_url==VIDEOS_WELDING[0]:
            frame_queue_list[0].put_nowait(frame)
        elif rtsp_url==VIDEOS_WELDING[1]:
            frame_queue_list[1].put_nowait(frame)
        elif rtsp_url==VIDEOS_WELDING[2]:
            frame_queue_list[2].put_nowait(frame)
        start_event.set()  
    cap.release()   

def save_image(welding_reset_imgs,results, step_name):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    url_path = f"{URL_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()

    cv2.imwrite(img_path, annotated_frame)
    welding_reset_imgs[step_name]=url_path

def process_video(model_path, video_source, start_event, stop_event,welding_reset_flag, welding_reset_imgs):
    # Load YOLO model
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} welding reset detection process is closing🛬")
            break
        
        if video_source.empty():
        # 队列为空，跳过处理
            continue
        frame = video_source.get()
        results = model.predict(frame,verbose=False,conf=0.4)
        for r in results:
            boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
            confidences = r.boxes.conf  # 提取所有检测到的置信度
            classes = r.boxes.cls  # 提取所有检测到的类别索引
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]
                if model_path == WEIGHTS_WELDING_RESET[0]:
                    if label== "machine_open":#检测焊机开关
                        welding_reset_flag[4] = True
                    if label=="machine_close":#检测焊机开关
                        welding_reset_flag[4] = False

                if model_path == WEIGHTS_WELDING_RESET[1]:
                    welding_reset_flag[0]=True#油桶默认没检测到时，就说明是安全的
                    if label=='oil_tank':#检测油桶,就说明油桶在危险区域
                        welding_reset_flag[0]=False
                    if label=='main_switch_open':
                        welding_reset_flag[1]=True
                    if label=='main_switch_close':
                        welding_reset_flag[1]=False

                if model_path == WEIGHTS_WELDING_RESET[2]:
                    welding_reset_flag[3]=False
                    if label=='weilding_componet':#检查焊件
                        welding_reset_flag[3]=True
                    if label=="grounding_wire" :
                        rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
                        WELDING_REGION3_shapely = Polygon(WELDING_REGION.tolist()) #shapely计算矩形检测框和多边形的iou使用
                        intersection = rect_shapely.intersection(WELDING_REGION3_shapely)
                                # 计算并集
                        union = rect_shapely.union(WELDING_REGION3_shapely)
                                # 计算 IoU
                        iou = intersection.area / union.area

                        if iou>0 :
                            welding_reset_flag[2]=True #表示搭铁线连接在焊台上
                        else:
                            welding_reset_flag[2]=False #表示未连接上


                if model_path == WEIGHTS_WELDING_RESET[0]:
                    if welding_reset_flag[4] and 'reset_step_5' not in welding_reset_imgs:
                        logging.info("welding machine switch is not reset")
                        save_image(welding_reset_imgs,results, "reset_step_5")

                if model_path == WEIGHTS_WELDING_RESET[1]:
                    if welding_reset_flag[1] and 'reset_step_2' not in welding_reset_imgs:
                        logging.info("main switch is not reset")
                        save_image(welding_reset_imgs,results, "reset_step_2")
                    if welding_reset_flag[0] and 'reset_step_1' not in welding_reset_imgs:
                        logging.info("oil tank is not reset")
                        save_image(welding_reset_imgs,results, "reset_step_1")
                if model_path == WEIGHTS_WELDING_RESET[2]:
                    if welding_reset_flag[2] and 'reset_step_3' not in welding_reset_imgs:
                        logging.info("grounding wire is not reset")
                        save_image(welding_reset_imgs,results, "reset_step_3")
                    if welding_reset_flag[3] and 'reset_step_4' not in welding_reset_imgs:
                        logging.info("weilding componet is not reset")
                        save_image(welding_reset_imgs,results, "reset_step_4")



        #运行到这里表示一个线程检测完毕
        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} welding reset detection process is running🚀")

    # 释放模型资源（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



