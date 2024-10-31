import cv2
import torch

from datetime import datetime
from ultralytics import YOLO
from utils import IoU_polygon
from config import SAVE_IMG_PATH_WELDING_K2,URL_IMG_PATH_WELDING_K2,WEIGHTS_WELDING_EXAM,VIDEOS_WELDING,WELDING_REGION1,WELDING_REGION2
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
            frame_queue_list[3].put_nowait(frame)
        start_event.set()  
    cap.release()   

def save_image(welding_exam_imgs,results, step_name,welding_exam_order):
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    url_path = f"{URL_IMG_PATH_WELDING_K2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    cv2.imwrite(img_path, annotated_frame)
    welding_exam_imgs[step_name]=url_path
    welding_exam_order.append(step_name)
    print(f"{step_name}完成")

# Function to process video with YOLO model
def process_video(model_path, video_source,start_event,stop_event,welding_exam_flag, welding_exam_imgs,welding_exam_order):
    # Load YOLO model
    model = YOLO(model_path)   
    while True:       
        if stop_event.is_set():
            print("复位子线程关闭")
            break

        if video_source.empty():
        # 队列为空，跳过处理
            continue
        
        frame = video_source.get()    

        results = model.predict(frame,verbose=False,conf=0.4)
        for r in results:
            if model_path==WEIGHTS_WELDING_EXAM[3]:
                if r.probs.top1conf>0.8:
                    label=model.names[r.probs.top1]
                    if label=='welding':
                        welding_exam_flag[6]=True
            else:            
                boxes = r.boxes.xyxy  
                confidences = r.boxes.conf  
                classes = r.boxes.cls  
                for i in range(len(boxes)):
                    cls = int(classes[i].item())
                    label = model.names[cls]
                    if model_path==WEIGHTS_WELDING_EXAM[0]:
                        if label== "machine_open":#检测焊机开关
                            welding_exam_flag[4] = True
                        if label=="machine_close" and welding_exam_flag[4]:#当打开过焊机开关，才能检测关闭状态
                            welding_exam_flag[8] = True
                    if model_path==WEIGHTS_WELDING_EXAM[1]:
                        oil_tank_flag=True
                        if label=="oil_tank":#检测油桶
                            oil_tank_flag=False#当检测到油桶时，说明危险源没有排除，所以为False
                        if label=="gloves":
                            welding_exam_flag[7]=True
                        if label=="main_switch_open":
                            welding_exam_flag[1]=True
                        if label=="main_switch_close" and welding_exam_flag[1]:
                            welding_exam_flag[12]=True

                        if not welding_exam_flag[0] and oil_tank_flag:
                            welding_exam_flag[0]=True

                    if model_path==WEIGHTS_WELDING_EXAM[2]:
                        grounding_wire_flag=False#搭铁线每次都要检测，初始化为False
                        welding_components_flag=False#焊件每次都要检测，初始化为False
                        
                        if label=="grounding_wire":
                            if IoU_polygon(boxes[i].tolist(), polygon_points=WELDING_REGION1.tolist())>0.1:
                                grounding_wire_flag=True
                            else:
                                grounding_wire_flag=False
                        if label=="welding_components":
                            if IoU_polygon(boxes[i].tolist(), polygon_points=WELDING_REGION2.tolist())>0.1:
                                #welding_exam_flag[3]=True
                                welding_components_flag=True
                        if label=="mask":
                            welding_exam_flag[5]=True
                        if label=="hamer":
                            welding_exam_flag[11]=True

                        if not welding_exam_flag[2] and grounding_wire_flag:
                            welding_exam_flag[2]=True#连接搭铁线
                        if not welding_exam_flag[9] and welding_exam_flag[2] and not grounding_wire_flag:
                            welding_exam_flag[9]=True#拆除搭铁线
                        if not welding_exam_flag[3] and welding_components_flag:
                            welding_exam_flag[3]=True#放置焊件
                        if not welding_exam_flag[10] and welding_exam_flag[3] and not welding_components_flag:
                            welding_exam_flag[10]=True#取下焊件
                        


                if model_path == WEIGHTS_WELDING_EXAM[0]:
                    if welding_exam_flag[4]=="open" and 'welding_exam_5' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_5",welding_exam_order)
                    if welding_exam_flag[8]=="close" and 'welding_exam_9' not in welding_exam_imgs and 'welding_exam_5' in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_9",welding_exam_order)

                if model_path == WEIGHTS_WELDING_EXAM[1]:
                    if welding_exam_flag[0] and 'welding_exam_1' not in welding_exam_imgs:#排除危险源
                        save_image(welding_exam_imgs,results,"welding_exam_1",welding_exam_order)
                    if welding_exam_flag[1] and 'welding_exam_2' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_2",welding_exam_order)
                    if welding_exam_flag[7]==True and 'welding_exam_8' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_8",welding_exam_order)
                    if welding_exam_flag[12] and 'welding_exam_13' not in welding_exam_imgs and 'welding_exam_2' in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_13",welding_exam_order)

                if model_path == WEIGHTS_WELDING_EXAM[2]:
                    if welding_exam_flag[2] and 'welding_exam_3' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_3",welding_exam_order)                       
                    if welding_exam_flag[9] and 'welding_exam_10' not in welding_exam_imgs and 'welding_exam_3' in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_10",welding_exam_order)        
                    if welding_exam_flag[3] and 'welding_exam_4' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_4",welding_exam_order)
                    if welding_exam_flag[5] and 'welding_exam_6' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_6",welding_exam_order)
                    if welding_exam_flag[10] and 'welding_exam_11' not in welding_exam_imgs and 'welding_exam_4' in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_11",welding_exam_order)
                    if welding_exam_flag[11] and 'welding_exam_12' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_12",welding_exam_order)

                if model_path == WEIGHTS_WELDING_EXAM[3]:
                    if welding_exam_flag[6]==True and 'welding_exam_7' not in welding_exam_imgs:
                        save_image(welding_exam_imgs,results,"welding_exam_7",welding_exam_order)            
            start_event.set()          


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    