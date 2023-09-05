import cv2
import numpy as np
import time

font = cv2.FONT_HERSHEY_SIMPLEX

class Boxing:
    #定义各种变量
    def __init__(self): 
        self.draw = False           #监测鼠标左键是否完成绘制
        self.pt0 = None             #矩形左上角坐标
        self.pt1 = None             #矩形右上角坐标
        self.rectangles = []        #存储的矩形坐标
        self.mid_gray = []          #存储的灰度中值
        self.end_time = None        #计时用
        self.timekey = None         #监测“s”键
        self.coorkey = None         #监测“c”键
        self.qua = []               #记录灰点象限
        self.n = []            
    
    #鼠标回调函数
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.pt0 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False
            self.pt1 = (x, y)
            self.rectangles.append((self.pt0, self.pt1))
        elif event == cv2.EVENT_MOUSEMOVE:
            self.pt1 = (x, y)
    
    #窗口1处理
    def _drawing_box(self, img):
        canvas = np.copy(img)
        
        #计算矩形框数据
        for i, (bpt0, bpt1) in enumerate(self.rectangles):
            cv2.rectangle(canvas, bpt0, bpt1, (255, 0, 0), 1)
            
            # 计算矩形范围内的灰度平均值
            roi = canvas[bpt0[1]:bpt1[1], bpt0[0]:bpt1[0]]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_gray = np.mean(gray_roi)
            
            # 添加矩形框编号和数据
            rect_text = f"Rectangle{i+1}:\n Loc: ({bpt0[0]}, {bpt0[1]})\n Width: {abs(bpt1[0] - bpt0[0])}\n Height: {abs(bpt1[1] - bpt0[1])}\n Avg Gray: {avg_gray:.2f}"
            cv2.putText(canvas, rect_text, (10, 18+(i+1)*18), font, 0.5, (0, 0, 255), 1)
        
            # 添加矩形框编号
            cv2.putText(canvas, f"{i+1}", (bpt0[0] - 10, bpt0[1] - 10), font, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, f"Number of Rectangles: {len(self.rectangles)}", (10, 18), font, 0.5, (255, 0, 0), 1)
        
        #绘制动画
        if self.draw:
            cv2.rectangle(canvas, self.pt0, self.pt1, (255, 0, 0), thickness=1)

        #计算灰度中值并输出
        if self.timekey:
            time_diff = time.time() -  self.end_time
            if time_diff <= 0:
                cv2.putText(canvas, "MidGray in calculation", (10, 18+(1+len(self.rectangles))*18), font, 0.5, (0, 255, 0), 1)
                self._calculate_mid_grays(canvas) #不断记录各矩形平均灰度值
            else:
                for i, mid_gray in enumerate(self.mid_gray):
                    cv2.putText(canvas, f"Rectangle{i+1}: MidGray: {np.median(mid_gray):.2f}", (10, 18+len(self.rectangles)*18+(i+1)*18), font, 0.5, (0, 255, 0), 1)
        
        #if self.coorkey:
        #    for number in range(0,(int)(len(self.rectangles)/2)):
        #        cv2.putText(canvas, f"Rectangle{i+1}: MidGray: {np.median(mid_gray):.2f}", (10, 18+len(self.rectangles)*18+(i+1)*18), font, 0.5, (0, 255, 0), 1)

        return canvas

    #窗口2处理
    def _drawing_calculate(self,img):
        canvas = np.copy(img)

        #设置画布
        canvas_size = 800
        axis_length = canvas_size // 2
        coor_canvas = 255 * np.ones((canvas_size, canvas_size), dtype=np.uint8)

        cv2.line(coor_canvas, (axis_length, 0), (axis_length, canvas_size), 0)
        cv2.line(coor_canvas, (0, axis_length), (canvas_size, axis_length), 0)

        for number in range(0,(int)(len(self.rectangles)/2)):
            gp = 0
            gq = 0
            p = 0
            q = 0
            
            for i in range(number*2,number*2+2):
                # 计算矩形范围内的灰度平均值
                (bpt0, bpt1) = self.rectangles[i]
                roi = canvas[bpt0[1]:bpt1[1], bpt0[0]:bpt1[0]]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                avg_gray = np.mean(gray_roi)
                if i%2:
                    q = avg_gray
                    gq = np.median(self.mid_gray[i])
                else:
                    p = avg_gray
                    gp = np.median(self.mid_gray[i])
        
            point_x = int((gp-p)*3 + axis_length)
            point_y = int((gq-q)*3 + axis_length)
            cv2.circle(coor_canvas, (point_x, point_y), 5, 0, -1)

            if len(self.qua[(int)(len(self.rectangles)/2)-1]):
                self._update_n(p, q, gp, gq, number)
            self.qua[number] = [self._get_quadrant(gp, gq, p, q)]
        print(self.n)
        return coor_canvas

    #记录各矩形框灰度值
    def _calculate_mid_grays(self, img):
        canvas = np.copy(img)
        for i,(bpt0, bpt1) in enumerate(self.rectangles):
            roi = canvas[bpt0[1]:bpt1[1], bpt0[0]:bpt1[0]]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_gray = np.mean(gray_roi)
            self.mid_gray[i].append(avg_gray)

    #对n进行处理
    def _update_n(self, p, q, gp, gq, number):
        prev_quadrant = self.qua[number][0]
        current_quadrant = self._get_quadrant(gp, gq, p, q)
        
        if prev_quadrant == 1 and current_quadrant == 4:
            self.n[number][0] += 0.25
        elif prev_quadrant == 2 and current_quadrant == 1:
            self.n[number][0] += 0.25
        elif prev_quadrant == 3 and current_quadrant == 2:
            self.n[number][0] += 0.25
        elif prev_quadrant == 4 and current_quadrant == 3:
            self.n[number][0] += 0.25
        elif prev_quadrant == 1 and current_quadrant == 2:
            self.n[number][0] -= 0.25
        elif prev_quadrant == 2 and current_quadrant == 3:
            self.n[number][0] -= 0.25
        elif prev_quadrant == 3 and current_quadrant == 4:
            self.n[number][0] -= 0.25
        elif prev_quadrant == 4 and current_quadrant == 1:
            self.n[number][0] -= 0.25


    #判定象限
    def _get_quadrant(self, x, y, p, q):
        if x >= p and y >= q:
            return 1
        elif x < p and y >= q:
            return 2
        elif x < p and y < q:
            return 3
        elif x >= p and y < q:
            return 4

    #启动函数
    def start(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Homeland")
        cv2.setMouseCallback("Homeland", self._on_mouse)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img1 = self._drawing_box(frame)
            cv2.imshow("Homeland", img1)
            if self.coorkey:
                img2 = self._drawing_calculate(frame)
                cv2.imshow("Coordinate System",img2)

            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('s'):
                self.end_time = time.time() + 5  # 其中“3”为监测时长 可调
                self.timekey = 1
                self.mid_gray = []
                for number in range(0,len(self.rectangles)):
                    self.mid_gray.append([])
                    
            elif cv2.waitKey(1) == ord('c'):
                self.coorkey = 1
                self.qua = []
                self.n = []
                for number in range(0,(int)(len(self.rectangles)/2)):
                    self.qua.append([])
                    self.n.append([0])
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    boxing = Boxing()
    boxing.start()