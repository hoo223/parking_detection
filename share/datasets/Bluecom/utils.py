from scipy.spatial import distance
import numpy as np
import cv2

class Utils:
    def __init__(self):
        pass
    
    # 각 행의 n 번째 열을 기준으로 sort (오름차순)
    # x : nd array
    # axis : 기준 열
    def sort_ndarray_by_col(self, x, axis=0):
        size = x.shape[0]
        col = [x[i][axis] for i in range(size)] # 기준 행의 원소들을 추출
        sorted_col = sorted(col) # 추출된 원소들을 정렬
        rank = [sorted_col.index(x[i][axis]) for i in range(size)] # 정렬된 index 추출
        #print(rank)
        try:
            sorted_array = [x[rank.index(i)] for i in range(size)]  # 정렬된 index 순서대로 정렬
        except: # 똑같은 순위가 있는 경우 예외처리
            overlap_rank = []
            for i in range(size):
                if rank.count(i) > 1:
                    overlap_rank.append(i)
            for r in overlap_rank:
                overlap_list = [i for i in range(size) if rank[i]==r]
                for k, idx in enumerate(overlap_list):
                    rank[idx] += k
            sorted_array = [x[rank.index(i)] for i in range(size)]  # 정렬된 index 순서대로 정렬

        return np.array(sorted_array)

    # 함수화
    def perspective_transform(self, src, point, height=256, width=256):    
        # 4 꼭지점 분류 
        sorted_by_y= self.sort_ndarray_by_col(point, 1)
        top_two = sorted_by_y[0:2]
        topLeft, topRight = self.sort_ndarray_by_col(top_two, 0)
        bottom_two = sorted_by_y[2:4]
        bottomLeft, bottomRight = self.sort_ndarray_by_col(bottom_two, 0)
        #print("tl:", topLeft)
        #print("bl", bottomLeft)
        #print("tr", topRight)
        #print("br", bottomRight)


        # 변환 전 4개 좌표 
        srcPoint = np.float32([topLeft, topRight, bottomRight , bottomLeft])

        # 변환 후 4개 좌표
        dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        # Perspective transformation
        matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
        dst = cv2.warpPerspective(src, matrix, (width, height))

        return dst

    # 이미지 상에 polygon 그리기
    def draw_label(self, img_path, points, save_path=""):
        '''
        image = img.imread(img_path)
        plt.imshow(image)

        ax = plt.gca()

        point = np.array([[data['p1'][0], data['p1'][1]],
                          [data['p2'][0], data['p2'][1]],
                          [data['p3'][0], data['p3'][1]],
                          [data['p4'][0], data['p4'][1]]])
        #print(point)

        polygon = patches.Polygon(point, fill=None ,edgecolor='k',ls='solid',lw=3)

        ax.add_patch(polygon)

        plt.show()
        '''
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        dst = cv2.polylines(src, np.int32([points]), True, (255, 0, 0), 6)

        return dst

    def extract_bb_points(self, points, width, height):
        if len(points.split(';')) == 4: # 라벨링된 point가 4개인 경우 = polygon 라벨링 (없음)
            ps = [(float(points.split(';')[0].split(',')[0]), float(points.split(';')[0].split(',')[1])),
                 (float(points.split(';')[1].split(',')[0]), float(points.split(';')[1].split(',')[1])),
                 (float(points.split(';')[2].split(',')[0]), float(points.split(';')[2].split(',')[1])),
                 (float(points.split(';')[3].split(',')[0]), float(points.split(';')[3].split(',')[1]))]

            # 4 꼭지점 분류 
            sorted_by_y= self.sort_ndarray_by_col(np.array(ps), 1)
            top_two = sorted_by_y[0:2]
            topLeft, topRight = self.sort_ndarray_by_col(top_two, 0)
            bottom_two = sorted_by_y[2:4]
            bottomLeft, bottomRight = list(self.sort_ndarray_by_col(bottom_two, 0))

        elif len(points.split(';')) == 2: # 라벨링된 point가 2개인 경우 = 직선 라벨링 (있음))
            ps = [(float(points.split(';')[0].split(',')[0]), float(points.split(';')[0].split(',')[1])),
                 (float(points.split(';')[1].split(',')[0]), float(points.split(';')[1].split(',')[1]))]

            # 2 꼭지점 분류
            if ps[0][0] < ps[1][0]:
                bottomLeft = ps[0]
                bottomRight = ps[1]
            else:
                bottomLeft = ps[1]
                bottomRight = ps[0]
        else:
            return None, 0, None

        print(len(points.split(';')))
        print('bb', bottomLeft, bottomRight) # bounding box 생성에 기준이 되는 주차면 앞 라인의 두 점

        # 라인으로부터 바운딩 박스 생성
        x_offset = 0
        width = distance.euclidean(bottomLeft, bottomRight)+x_offset
        y_offset = width/10.0
        height = -width*2/3.0
        #print(ps, width, height)


        b1 = (bottomLeft[0], bottomLeft[1]+y_offset)
        b2 = (bottomRight[0], bottomRight[1]+y_offset)
        b3 = (b2[0], b2[1]+height)
        b4 = (b1[0], b1[1]+height)
        print('ps:',np.array([b1, b2, b3, b4]), width, np.array(ps))

        return np.array([b1, b2, b3, b4]), width, np.array(ps)

    def get_occupied_info(self, polyline, width, height):
        label = polyline.get('label')
        points = polyline.get('points') # 2개
        bb_points, width, _ = self.extract_bb_points(points, width, height)
        return label, width, bb_points

    def get_empty_info(self, polygon, width, height):
        label = polygon.get('label')
        points = polygon.get('points') # 4개 or 5개
        bb_points, width, poly_points = self.extract_bb_points(points, width, height)
        empty_type_dict = {'일반형':0, '경형':1, '장애인전용':2, '여성우선주차장':3, '환경친화적 자동차 전용':4, 'etc':5, '기타':5}
        empty_type = empty_type_dict[polygon.find('attribute').text]
        return label, width, bb_points, poly_points, empty_type

    def get_obstacle_info(self, polygon, width, height):
        label = polygon.get('label')
        points = polygon.get('points')
        bb_points, width, poly_points = self.extract_bb_points(points, width, height)
        obstacle_type_dict = {'사람':0, '이륜차':1, '기타':2}
        obs_type = obstacle_type_dict[polygon.find('attribute').text]
        return label, width, bb_points, poly_points, obs_type