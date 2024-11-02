import pdb
import glob
import cv2
import os
import numpy as np
    
class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_features_and_keypoints_using_SIFT(self, image):
        # detect and extract features from the image
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(image, None)
        keypoints = np.float32([i.pt for i in keypoints])
        return keypoints, features
    
    def get_valid_matches(self,featuresA,featuresB,lowe_ratio):
        #initialize a matcher
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = matcher.knnMatch(featuresA, featuresB, 2)
        valid_matches = []
        #using lowe ratio to find all the valid matches
        for val in All_Matches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))
        return valid_matches
    
    def find_homography_matrix(self, src_pts, dst_pts):
        #initializing A matrix which has 9 cols
        A = []
        for i in range(4):
            x, y = src_pts[i][0], src_pts[i][1]
            u, v = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        # Solving for H using SVD
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        # Normalizing H to make the bottom-right element 1
        return H / H[2, 2]
    
    def apply_homography_transformation(self, H, point):
        x, y = point[0], point[1]
        transformed_point = np.dot(H, np.array([x, y, 1]))
        transformed_point /= transformed_point[2]
        return transformed_point[:2]
    
    def compute_homography_using_RANSAC(self, pointsA, pointsB, max_Threshold, max_iterations=1000):
        #initializing the variables
        best_H = None
        best_inliers_count = 0
        n_points = len(pointsA)
        
        #RANSAC 
        for _ in range(max_iterations):
            #Randomly select 4 corresponding points
            idx = np.random.choice(n_points, 4, replace=False)
            source = pointsA[idx]
            dest = pointsB[idx]

            #Computing homography matrix H
            H = self.find_homography_matrix(source, dest)

            #Counting inliers by applying H to all points and checking the reprojection error
            inliers_count = 0
            for i in range(n_points):
                projected_point = self.apply_homography_transformation(H, pointsA[i])
                distance = np.linalg.norm(projected_point - pointsB[i])
                
                if distance < max_Threshold:
                    inliers_count += 1

            #If the inlier count exceed the previous value, update the variables
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_H = H

        return best_H
    
    def match_keypoints(self, KeypointsA, KeypointsB, featuresA, featuresB, lowe_ratio, max_Threshold):
        valid_matches = self.get_valid_matches(featuresA,featuresB,lowe_ratio)

        if len(valid_matches) <= 4:
            return None

        points_A = np.float32([KeypointsA[i] for (_, i) in valid_matches])
        points_B = np.float32([KeypointsB[i] for (i, _) in valid_matches])

        #computing the homography matrix
        homograpy = self.compute_homography_using_RANSAC(points_A, points_B, max_Threshold)
        return homograpy
    
    def get_warp_perspective(self, imageA, imageB, Homography):
        #Calculating dimensions of the output image
        heightA, widthA = imageA.shape[:2]
        heightB, widthB = imageB.shape[:2]
        output_width = widthA + widthB
        output_height = max(heightA, heightB)
        
        #Initializing the output image
        result_image = np.zeros((output_height, output_width, 3), dtype=imageA.dtype)
        
        #Placing imageB directly on the output canvas at (0, 0)
        result_image[:heightB, :widthB] = imageB

        # Calculating the inverse homography matrix
        H_inv = np.linalg.inv(Homography)
        
        # Iterate over each pixel in the output image
        for y in range(output_height):
            for x in range(output_width):
                # Mapping (x, y) in the output image back to imageA
                dest_pixel = np.array([x, y, 1])
                src_pixel = H_inv @ dest_pixel
                #normalizing
                src_pixel /= src_pixel[2]  
                
                #Get source coordinates
                src_x, src_y = src_pixel[:2]

                #Check if the source coordinates are within bounds of imageA
                if 0 <= src_x < widthA and 0 <= src_y < heightA:
                    #Perform bilinear interpolation to avoid gaps
                    x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                    x1, y1 = min(x0 + 1, widthA - 1), min(y0 + 1, heightA - 1)
                    
                    #Calculate interpolation weights
                    a, b = src_x - x0, src_y - y0
                    pixel_value = (
                        (1 - a) * (1 - b) * imageA[y0, x0] +
                        a * (1 - b) * imageA[y0, x1] +
                        (1 - a) * b * imageA[y1, x0] +
                        a * b * imageA[y1, x1]
                    )
                    
                    #Place interpolated pixel value in the result image
                    result_image[y, x] = pixel_value

        return result_image


    def image_stitcher(self, imageB,imageA, lowe_ratio=0.75, max_Threshold=4.0):
        # detect the features and keypoints from sift
        key_points_A, features_of_A = self.detect_features_and_keypoints_using_SIFT(imageA)
        key_points_B, features_of_B = self.detect_features_and_keypoints_using_SIFT(imageB)

        # get the valid matched points and compute H matrix
        Homography = self.match_keypoints(key_points_A, key_points_B, features_of_A, features_of_B, lowe_ratio, max_Threshold)
        if Homography is None:
            return None

        #computing the warped perspective of image using computed homography matrix
        result_image = self.get_warp_perspective(imageA, imageB, Homography)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result_image

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        homography_matrix_list =[]
        no_of_images = len(all_images)
        images = []
        for i in range(no_of_images):
            temp = cv2.imread(all_images[i])
            temp = cv2.resize(temp,(500,500))
            images.append(temp)

        result = self.image_stitcher(images[no_of_images - 2], images[no_of_images - 1])
        for i in range(no_of_images-2):
            result = self.image_stitcher(images[no_of_images - i - 3], result)
        #final_image = cv2.cvtColor(final_image,cv2.COLOR_GRAY2RGB)

        return result,homography_matrix_list


