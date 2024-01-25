import json

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytesseract
import openpyxl
import os
import threading
import time

class SimpleRequestHandler(BaseHTTPRequestHandler):
    folder_path = 'new'
    pathRotated = None
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    answer_sheet2 = {1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B', 6: 'B', 7: 'A', 8: 'D', 9: 'B', 10: 'C', 11: 'D', 12: 'B',
                    13: 'C', 14: 'D', 15: 'B', 16: 'C', 17: 'E', 18: 'C', 19: 'A', 20: 'A', 21: 'C', 22: 'A', 23: 'C',
                    24: 'C', 25: 'C', 26: 'B', 27: 'C', 28: 'C', 29: 'A', 30: 'C', 31: 'B', 32: 'B', 33: 'A', 34: 'E',
                    35: 'C', 36: 'D', 37: 'B', 38: 'C', 39: 'C', 40: 'B', 41: 'C', 42: 'D', 43: 'C', 44: 'A', 45: 'B',
                    46: 'C', 47: 'B', 48: 'B', 49: 'C', 50: 'A'}
    mapping_key = {'F': 'A', 'G': 'B', 'H': 'C', 'I': 'D', 'J': 'E'}
    answer_sheet = None;

    multiply_shaded = 0
    correctly_answered = 0
    wrong = 0
    nameImage = None
    objects_list = []

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def process_images(self):
        if os.path.exists(self.folder_path):
            grade_data = []
            tracings = []

            for filename in os.listdir(self.folder_path):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%filename",filename)
                try:
                    file_path = os.path.join(self.folder_path, filename)

                    if os.path.isfile(file_path) and file_path.lower().endswith(('.jpeg', '.png', 'jpg')):
                        print('\n\n\n', filename, '\n\n\n')
                        warp = self.transform(file_path)
                        eroded = self.erode_warp(warp)
                        columnar_Assignment, row_Assignment, shaded_circles, warp, image = self.detect_circles(
                            eroded, warp)
                        shaded_mapped = self.trace_zletter(columnar_Assignment, row_Assignment, shaded_circles)
                        correctly_answered, multiply_answered, wrong, name = self.mark_ztest(shaded_mapped,
                                                                                       self.answer_sheet,
                                                                                       self.mapping_key,
                                                                                       filename)
                        grade_data.append((filename, correctly_answered, multiply_answered, wrong))
                        # self.objects_list.append({correctly_answered,multiply_answered,wrong,name})
                        self.objects_list.append({
                            'multiply_shaded': multiply_answered,
                            'correctly_answered': correctly_answered,
                            'wrong': wrong,
                            'name': name
                        })
                        tracings.append(warp)
                    else:
                        print(f"Skipping non-image file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue

            self.generate_excel(grade_data)
        else:
            print(f"The folder '{self.folder_path}' does not exist.")

        for idx, warp in enumerate(tracings):
            cv2.imshow(f'image{idx}', warp)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    def do_OPTIONS(self):
        # Handle pre-flight CORS request
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        # Handle CORS for the actual request
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Parse the JSON data
        received_data = json.loads(post_data.decode('utf-8'))

        # Access the 'answers' property
        answers_data = received_data.get('answers', {})

        # Create the desired format without single quotes around the numbers
        formatted_data = {int(key): value for key, value in answers_data.items()}

        # Your processing logic here
        # For simplicity, we'll just echo the received data
        response_data = {'result': 'Server received data', 'data': formatted_data}
        print("#######################", response_data)

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

        # Assign the formatted data to the variable
        SimpleRequestHandler.answer_sheet = formatted_data
        print("#############answer_sheet2:", SimpleRequestHandler.answer_sheet)

        # Process the images using the method within SimpleRequestHandler
        self.process_images()

    def get_test_results(self):
        """
        Retrieve the results of the test.
        """
        # result_data = {
        #     'multiply_shaded': self.multiply_shaded,
        #     'correctly_answered': self.correctly_answered,
        #     'wrong': self.wrong,
        #     'name': self.nameImage
        #
        # }

        result_data = self.objects_list;
        print("##############result_data", result_data)  # Add this line for debugging
        return result_data

    def do_GET(self):
        """
        Handle GET requests.
        """
        # Handle CORS for the actual request
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if self.path == '/get_test_results':
            result_data = self.get_test_results()
            self.wfile.write(json.dumps(result_data).encode('utf-8'))
        elif self.path == '/':  # Ignore the '/' path, it's likely an OPTIONS request
            pass
        else:
            print("Invalid path: ", self.path)



    def transform(self,name):
        def sort_contour(rect, edges):
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            i = 0
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)
            for contour in sorted_contours:
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.04 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3:
                    j = 0
                    for rectc in rect:
                        for tric in approx:
                            if abs(tric[0][0] - rectc[0][0]) > 120 or abs(tric[0][1] - rectc[0][1]) > 120:
                                break
                        else:
                            # print('triangle found')
                            nc = np.array([rect[j-i] for i in range(4)])
                            return nc
                        j += 1
                i += 1
            else:
                # print('triangle not found')
                return sort_contour_clockwise(rect)

        def sort_contour_clockwise(contour):
            centroid = np.mean(contour, axis=0)
            angles = np.arctan2(contour[:,0,1] - centroid[0,1], contour[:,0,0] - centroid[0,0])
            sorted_indices = np.argsort(angles)
            sorted_contour = contour[sorted_indices]
            return sorted_contour

        image = cv2.imread(name)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        retval, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        dilated = cv2.erode(gray, np.ones((5, 5)), iterations=1)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            return None

        newapprox = sort_contour(approx, edges)
        x, y, w, h = cv2.boundingRect(approx)
        #######
        # the if block swaps the width and height if there is a 90 degree rotation
        #######
        if w>h:
            t = w
            w = h
            h = t
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        corners = np.float32([point[0] for point in newapprox])
        width, height = w, h
        target_corners = np.float32([[-5, -5], [width+5, -5], [width+5, height+5], [-5, height+15]])

        perspective_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))
        return transformed

    def rotate_image(self,angle, path):
        image = cv2.imread(path)

        # Get image dimensions
        height, width = image.shape[:2]
        print("Original Height and Width:", height, width)

        # Define the pivot point (center of the image)
        pivot = (width // 2, height // 2)

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(pivot, -angle, 1.0)

        # Apply the image rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Find the bounding box of the rotated image
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])

        new_width = int((width * cos_angle) + (height * sin_angle))
        new_height = int((width * sin_angle) + (height * cos_angle))

        # Adjust the rotation matrix based on the bounding box
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2

        # Apply the corrected rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return rotated_image

    def align_perspective(self,path):
        def get_distance_from_origin(point):
            return point[0]**2 + point[1]**2

        def order_points_in_counter_clockwise(points):
            # Convert the input list to a NumPy array
            points = np.array(points)

            # Calculate distances from origin for each point
            distances = np.sum(points**2, axis=1)

            # Find the index of the point closest to the origin
            nearest_point_index = np.argmin(distances)

            # Reorder points starting from the nearest point in counter-clockwise direction
            ordered_points = [points[nearest_point_index]]
            remaining_points = np.delete(points, nearest_point_index, axis=0)

            # Sort the remaining points based on the angle they make with the positive y-axis
            ordered_points = np.concatenate((ordered_points, sorted(remaining_points, key=lambda point: np.arctan2(point[0], point[1]))))

            return ordered_points


        gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and help edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 100)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the farthest points of the largest contour
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        farthest_points = np.squeeze(approx, axis=1)
        farthest_points = order_points_in_counter_clockwise(farthest_points)
        print('farthest points: ', farthest_points)

        # Set the dimention of the reference rectangle
        width = path.shape[1]
        height = path.shape[0]

        src_pts = farthest_points.astype(np.float32)
        dst_pts = np.array([[0,0], [0, height], [width, height], [width, 0]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("%%%%%%%%%%%%%%%inside M",M)
        # Apply the perspective transform to correct for small rotations and alignment
        warped = cv2.warpPerspective(path, M, (width, height))

        return warped

    def erode_warp(self,path):
        #image = cv2.imread(path)
        gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

        THRESHOLD = 128

        # Apply thresholding using NumPy
        thresholded_image = np.where(gray > THRESHOLD, 255, 0)

        # Convert back to 3 channels for display
        thresholded_image = cv2.cvtColor(thresholded_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Define a kernel (structuring element)
        kernel = np.ones((2, 2), np.uint8)

        # Perform erosion using cv2.erode()
        erosion_result = cv2.erode(thresholded_image, kernel, iterations=1)


        # Apply dilation using cv2.dilate()
        dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)

        # Apply closing using cv2.morphologyEx() with MORPH_CLOSE operation
        closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

        return erosion_result

    def mark_ztest(self,Answers, answer_sheet, key, name):
        for i in key:
            for j in Answers:
                if j[1]==i:
                    index = Answers.index(j)
                    temp = (j[0]+25, key[i])
                    Answers[index] = temp
        Answers =  sorted(Answers, key=lambda x:x[0])

        multiply_shaded=0
        correctly_answered=0
        student_mark = {}
        for k in Answers:
            if k[0] not in student_mark:
                student_mark[k[0]]=k[1]
            else:
                student_mark[k[0]]='REPEATED'

        for kk in student_mark:
            if student_mark[kk]==answer_sheet[kk]:
                correctly_answered+=1
            elif student_mark[kk]=='REPEATED':
                multiply_shaded+=1
        print('\nStudent Answered as follows:')
        for jj in student_mark:
            print(str(jj)+': '+student_mark[jj])

        #print('paper: ', student_mark)
        print('\nmultiply shaded questions: ', multiply_shaded)
        print('correctly answered questions: ', correctly_answered)
        wrong = len(answer_sheet)-(multiply_shaded+correctly_answered)
        print('wrong: ', wrong)

        SimpleRequestHandler.multiply_shaded = multiply_shaded
        SimpleRequestHandler.correctly_answered = correctly_answered
        SimpleRequestHandler.wrong = wrong
        SimpleRequestHandler.nameImage = name

        return [correctly_answered, multiply_shaded, wrong, name]

    def generate_excel(self,data):
        # Create a new workbook and select the active sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Add data to the sheet
        sheet['A1'] = 'Name'
        sheet['B1'] = 'correctly_answered'
        sheet['C1'] = 'multiply_answered'
        sheet['D1'] = 'wrong'

        for row_index, (name, correct, repeated, wrong) in enumerate(data, start=2):
            sheet[f'A{row_index}'] = name
            sheet[f'B{row_index}'] = correct
            sheet[f'C{row_index}'] = repeated
            sheet[f'D{row_index}'] = wrong

        # Save the workbook to a file
        workbook.save('grade.xlsx')
        print('\n\nexcel file successfully generated.')

    def trace_zletter(self,x_groups, y_groups, shaded):
        for key in x_groups:
            x_groups[key] = round(x_groups[key],1)
        for key in y_groups:
            y_groups[key] = round(y_groups[key],1)

        def find_closest_value(target, values):
            closest = min(values, key=lambda x: abs(x - target))
            return closest

        shaded_mapped = []

        for point in shaded:
            x_key = find_closest_value(point[0], x_groups.values())
            y_key = find_closest_value(point[1], y_groups.values())
            x_mapped = [key for key, value in x_groups.items() if value == x_key][0]
            y_mapped = [key for key, value in y_groups.items() if value == y_key][0]
            shaded_mapped.append((y_mapped, x_mapped))

        shaded_mapped = sorted(shaded_mapped, key=lambda x:x[0])

        return shaded_mapped

    def detect_circles(self,path, warp):
        shaded_circles = []
        image = path
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 100, apertureSize=3)
        # Apply Hough Circle Transform

        #####
        # modified circle detection
        #####

        #circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=15, param1=26, param2=30, minRadius=7, maxRadius=13)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=19, param1=26, param2=20, minRadius=7, maxRadius=13)

        # Draw circles on the original image
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                # Extract the region of interest (ROI) for the circle
                circle_roi = image[i[1] - radius+5:i[1] + radius-5, i[0] - radius+5:i[0] + radius-5]

                # Calculate average intensity of the circle
                average_intensity = np.mean(circle_roi)

                # Calculate bounding box coordinates
                #####
                # modified circle bounding rect calculation
                #####
                #x, y, w, h = i[0] - radius, i[1] - radius, 2 * radius, 2 * radius
                x, y, w, h = i[0] - radius+3, i[1] - radius+3, 2 * radius-6, 2 * radius-6

                # Draw the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Mark the circles in green
                cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.circle(image, center, radius, (0, 255, 0), 2)

                # Determine if the circle is shaded based on average intensity
                if average_intensity < 100:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Mark filled circles in red
                    cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    shaded_circles.append(center)


            # Sort circles based on y-coordinates
            circlesY = circles[0, circles[0, :, 1].argsort()]
            circlesX = circles[0, circles[0, :, 0].argsort()]

            x_groups = {}
            y_grouping_range=15
            x_grouping_range=15

            # Process circles and group by x-coordinates
            for kk in circlesX:
                x, y, r = kk

                # Find the group key within the specified range
                group_key = next((key for key in x_groups if abs(x - key) <= x_grouping_range), None)

                if group_key is not None:
                    # Add the circle to the existing group
                    x_groups[group_key].append((x, y, r))
                else:
                    # Create a new group
                    x_groups[x] = [(x, y, r)]


            # Count the number of x-coordinate groups
            num_x_groups = len(x_groups)
            print(f"Number of distinct x-coordinate groups: {num_x_groups}")

            # Determine the pattern for assigning letters
            letter_pattern = "ABCDEFGHIJ" if num_x_groups == 10 else "ABCDEFGH"
            columnar_Assignment = {}
            jj=0

            # sorted_x_groups = sorted(x_groups.items(), key=lambda item: np.mean([circle[0] for circle in item[1]]))

            for x, group in x_groups.items():
                x_average = np.mean([circle[0] for circle in group])
                columnar_Assignment[letter_pattern[jj]]=x_average
                jj+=1
            print('\nx-groups:')
            for key in columnar_Assignment:
                print(key+': '+str(columnar_Assignment[key]))

            # Calculate the average distance among half x-coordinate groups
            x_centers = sorted(list(x_groups.keys()))
            distances = [x_centers[i+1] - x_centers[i] for i in range((num_x_groups//2)-1)]
            avg_Xdistance = np.mean(distances)
            print(f"Average distance among x-coordinate first column groups: {avg_Xdistance}")

            y_groups = {}

            # Process circles and group by y-coordinates
            for jj in circlesY:
                x, y, r = jj
                # Find the group key within the specified range
                group_key = next((key for key in y_groups if abs(y - key) <= y_grouping_range), None)

                if group_key is not None:
                    # Add the circle to the existing group
                    y_groups[group_key].append((x, y, r))
                else:
                    # Create a new group
                    y_groups[y] = [(x, y, r)]

            # Count the number of y-coordinate groups
            num_y_groups = len(y_groups)
            print(f"Number of distinct y-coordinate groups: {num_y_groups}")

            # sorted_y_groups = sorted(y_groups.items(), key=lambda item: np.mean([circle[1] for circle in item[1]]))

            row_Assignment = {}
            ii=1

            # Print information about each y-coordinate group and its average
            for y, group in y_groups.items():
                # Sort circles within the y-coordinate group based on x-coordinates
                group.sort(key=lambda circle: circle[0])
                #avg_x = np.mean([circle[0] for circle in group])
                avg_y = np.mean([circle[1] for circle in group])
                row_Assignment[ii]=avg_y
                # print(f"Y-coordinate group {y} (Group Number: {ii}, Average Y-coordinate: {avg_y}):")
                ii+=1

            print('\ny-groups:')
            for key in row_Assignment:
                print(str(key)+': '+str(row_Assignment[key]))

            # Calculate the average distance among y-coordinate groups
            y_centers = sorted(list(y_groups.keys()))
            distances = [y_centers[i+1] - y_centers[i] for i in range(num_y_groups-1)]
            avg_Ydistance = np.mean(distances)
            print(f"Average distance among y-coordinate groups: {avg_Ydistance}")

            the_first_element = [columnar_Assignment['A'], row_Assignment[1]]
            print('first element: ', the_first_element)


        return [columnar_Assignment, row_Assignment, shaded_circles, warp, image]

    def find_text_orientation(self,image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use the orientation information in the OCR result
        orientation = pytesseract.image_to_osd(gray)
        lines = orientation.splitlines()
        #print("#############inside lines",lines)
        text = pytesseract.image_to_string(gray)

        # Extract the rotation angle from the orientation result
        rotation_angle = 0.0
        for line in lines:
            if line.startswith('Rotate:'):
                rotation_angle = float(line.split(':')[1])

        return rotation_angle

def run_server(port=8082):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleRequestHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()


def run_server_and_process():

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    time.sleep(2)



if __name__ == "__main__":
    run_server()

