from skimage import io, img_as_float
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.morphology import disk, closing, opening, remove_small_objects, remove_small_holes
from skimage.filters import median, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, canny
from scipy import ndimage
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
import os

class StrawberryBoundingBoxClassifier:
    def __init__(self, dataset_path="data"):
        self.dataset_path = dataset_path
        self.image_files = sorted(glob.glob(os.path.join(dataset_path, "*.png"))) + \
                           sorted(glob.glob(os.path.join(dataset_path, "*.jpg"))) + \
                           sorted(glob.glob(os.path.join(dataset_path, "*.jpeg")))
        self.current_index = 0

        if not self.image_files:
            raise RuntimeError(f"No images found in {dataset_path}")
        
        self.ws_params = {
            'use_watershed': True,
            'min_distance': 15,
            'threshold': 0.03
        }

        # OTSU parameters
        self.classes = 3
        self.target_region_index = 2
        self.disk_radius = 6

        # LAB threshold for ripe (red) strawberries
        self.lab_thresh_a = 10

        # HSV thresholds for unripe (yellowish/greenish) strawberries
        self.hue_min = 45
        self.hue_max = 85
        self.sat_min = 20/255.0
        self.val_min = 150/255.0

        # Classification thresholds
        self.classification_thresholds = {
            'maturing_red_min': 12,   
            'maturing_red_max': 25,   
            'ripe_red_min': 25,       
            'unripe_hue_min': 45,
            'unripe_hue_max': 85,
            'unripe_sat_min': 0.3,
        }

        # Results storage
        self.bounding_boxes = []
        self.areas = []
        self.classifications = []
        self.avg_colors = []

        # figure with subplots: original, mask, result, and classification
        self.fig, ((self.ax_orig, self.ax_mask), (self.ax_res, self.ax_class)) = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Strawberry Segmentation and Classification', fontsize=16)

        self.setup_buttons()
        self.process_current_image()

    def setup_buttons(self):
        ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])
        ax_next = plt.axes([0.25, 0.01, 0.1, 0.05])
        ax_info = plt.axes([0.4, 0.01, 0.3, 0.05])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_info = Button(ax_info, f'Image {self.current_index+1}/{len(self.image_files)}')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.process_current_image()
            self.update_info_button()

    def next_image(self, event):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.process_current_image()
            self.update_info_button()

    def update_info_button(self):
        fn = os.path.basename(self.image_files[self.current_index])
        self.btn_info.label.set_text(f'{self.current_index+1}/{len(self.image_files)} - {fn}')
        self.fig.canvas.draw()

    def apply_watershed(self, binary_image, params):

        if not params['use_watershed']:
            return binary_image

        distance = ndimage.distance_transform_edt(binary_image)
        min_distance = max(1, params['min_distance'])
        threshold_abs = params['threshold'] * distance.max()

        local_maxima = peak_local_max(distance,
                                      labels=binary_image,
                                      min_distance=min_distance,
                                      threshold_abs=threshold_abs)

        if len(local_maxima) == 0:
            return binary_image

        markers = np.zeros_like(binary_image, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima, start=1):
            markers[y, x] = i

        labels = watershed(-distance, markers, mask=binary_image.astype(bool))
        return (labels > 0)
    
    def filter_regions_by_shape(self, labels, gray_image,
                            edge_density_thresh=0.8,
                            min_area=100,
                            max_area=90000,
                            min_aspect_ratio=0.2,
                            max_aspect_ratio=4,
                            min_circularity=0.05,
                            min_solidity=0.3):
   
        edges = canny(gray_image)
        mask = np.zeros_like(labels, dtype=bool)

        for prop in regionprops(labels):
            area = prop.area
            if area < min_area or area > max_area:
                continue

            minr, minc, maxr, maxc = prop.bbox
            height = maxr - minr
            width  = maxc - minc
            ar = width / height if height > 0 else 0
            if ar < min_aspect_ratio or ar > max_aspect_ratio:
                continue

            if prop.solidity < min_solidity:
                continue

            per = prop.perimeter if prop.perimeter > 0 else 1
            circ = 4 * math.pi * area / (per * per)
            if circ < min_circularity:
                continue

            ed = np.sum(edges[labels == prop.label]) / area
            if ed > edge_density_thresh:
                continue

            mask[labels == prop.label] = True

        return mask

    def extract_bounding_boxes_from_mask(self, mask):
        # Label connected components
        labeled_mask = label(mask)
        
        bounding_boxes = []
        areas = []

        # Get region properties
        for region in regionprops(labeled_mask):
            if region.area > 1000:  # minimum area threshold
                # Get bounding box coordinates (minr, minc, maxr, maxc)
                minr, minc, maxr, maxc = region.bbox
                
                bounding_boxes.append({
                    'minr': minr,
                    'minc': minc, 
                    'maxr': maxr,
                    'maxc': maxc,
                    'width': maxc - minc,
                    'height': maxr - minr,
                    'centroid': region.centroid
                })
                areas.append(region.area)
        
        return bounding_boxes, areas

    def calculate_average_color_in_bbox(self, image, bbox):
        # Extract region from bounding box
        minr, minc, maxr, maxc = bbox['minr'], bbox['minc'], bbox['maxr'], bbox['maxc']
        
        # Ensure coordinates are within image bounds
        minr = max(0, minr)
        minc = max(0, minc)
        maxr = min(image.shape[0], maxr)
        maxc = min(image.shape[1], maxc)
        
        # Extract region
        region = image[minr:maxr, minc:maxc]
        
        if region.size == 0:
            return None
            
        # Calculate average RGB
        avg_rgb = np.mean(region.reshape(-1, 3), axis=0)
        
        # Create a small patch with the average color for conversion
        color_patch = np.full((10, 10, 3), avg_rgb/255.0, dtype=np.float64)
        
        # Convert to HSV
        hsv_patch = rgb2hsv(color_patch)
        avg_hsv = [hsv_patch[0, 0, 0] * 360, hsv_patch[0, 0, 1], hsv_patch[0, 0, 2]]
        
        # Convert to LAB
        lab_patch = rgb2lab(color_patch)
        avg_lab = [lab_patch[0, 0, 0], lab_patch[0, 0, 1], lab_patch[0, 0, 2]]
        
        return {
            'rgb': avg_rgb,
            'hsv': avg_hsv,
            'lab': avg_lab
        }

    def classify_strawberry(self, avg_colors):
        if avg_colors is None:
            return 'Unknown'
        
        hsv = avg_colors['hsv']
        lab = avg_colors['lab']
        
        # Classification logic
        if (self.classification_thresholds['maturing_red_min']
            <= lab[1] < self.classification_thresholds['maturing_red_max']):
            return 'Maturing'

        # Ripe strawberries: high a* value in LAB (red)
        if lab[1] > self.classification_thresholds['ripe_red_min']:
            return 'Ripe'
        
        # Unripe strawberries: green/yellow hues in HSV
        elif (self.classification_thresholds['unripe_hue_min'] <= hsv[0] <= 
              self.classification_thresholds['unripe_hue_max'] and 
              hsv[1] > self.classification_thresholds['unripe_sat_min']):
            return 'Unripe'
        
        # Intermediate or uncertain
        else:
            return 'Unknown'

    def get_classification_color(self, classification):
        colors = {
            'Ripe': (0, 255, 0),        # Red
            'Unripe': (255, 0, 0),      # Green
            'Maturing': (255, 255, 0),  # Yellow
            'Unknown': (128, 128, 128)  # Gray
        }
        return colors.get(classification, (128, 128, 128))
    
    def get_adaptive_watershed_params(self, binary_image):
    # Calculate average object size to adapt parameters
        labeled = label(binary_image)
        regions = regionprops(labeled)
    
        if not regions:
            return self.ws_params
    
        # Get average area and adjust min_distance accordingly
        avg_area = np.mean([r.area for r in regions])
        estimated_radius = np.sqrt(avg_area / np.pi)
    
        # Adaptive parameters
        adaptive_params = {
            'use_watershed': True,
            'min_distance': max(15, int(estimated_radius * 0.6)),  # Smaller min_distance for better separation
            'threshold': 0.05  # Lower threshold to detect more peaks
        }
    
        return adaptive_params
    
    def preprocess_for_separation(self, mask):
        from skimage.morphology import erosion, dilation, disk, diamond
    
        # Light erosion to separate touching objects
        eroded = erosion(mask, disk(5))
    
        # Remove very small objects that might be noise
        eroded = remove_small_objects(eroded, min_size=200)
    
        # Dilate back to restore size
        restored = dilation(eroded, disk(3))
    
        return restored

    def process_current_image(self):
        current_file = self.image_files[self.current_index]
        print(f"Processing: {current_file}")

        img_rgb = io.imread(current_file)
        img_float = img_as_float(img_rgb)

        hsv = rgb2hsv(img_float)
        V = hsv[:, :, 2]
        V_scaled = (V * 255).astype(np.uint8)
        selem = disk(self.disk_radius)
        filtered_V = median(V_scaled, selem)

        thresholds = threshold_multiotsu(filtered_V, classes=self.classes)
        regions = np.digitize(filtered_V, bins=thresholds)
        mask_otsu = (regions == self.target_region_index)

        lab = rgb2lab(img_float)
        A = lab[:, :, 1]
        mask_lab_ripe = (A > self.lab_thresh_a) 

        h = hsv[:, :, 0] * 360
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask_hsv_unripe = (
            (h >= self.hue_min) & (h <= self.hue_max) &
            (s >= self.sat_min) & (v >= self.val_min)
        )

        mask_color = mask_lab_ripe | mask_hsv_unripe
        mask_refined = mask_otsu & mask_color

        mask_refined = closing(mask_refined, disk(5))
        mask_refined = opening(mask_refined, disk(3))
        mask_refined = remove_small_objects(mask_refined, min_size=500)
        mask_refined = remove_small_holes(mask_refined, area_threshold=1000)

        h_green = hsv[:, :, 0] * 360
        s_green = hsv[:, :, 1]
        v_green = hsv[:, :, 2]

        h_min_green, s_min_green, v_min_green = 30, 20/255, 40/255
        h_max_green, s_max_green, v_max_green = 115, 255/255, 200/255

        mask_green = (
            (h_green >= h_min_green) & (h_green <= h_max_green) &
            (s_green >= s_min_green) & (s_green <= s_max_green) &
            (v_green >= v_min_green) & (v_green <= v_max_green))
        
        mask_refined = self.preprocess_for_separation(mask_refined)

        mask_nogreen = mask_refined & (~mask_green)

        mask_ws = self.apply_watershed(mask_nogreen, self.ws_params)

        gray = rgb2gray(img_float)
        labels_ws = label(mask_ws)

        mask_shape = self.filter_regions_by_shape(
            labels=labels_ws,
            gray_image=gray,
            edge_density_thresh=0.55,
            min_area=2500,
            max_area=90000,
            min_aspect_ratio=0.2,
            max_aspect_ratio=4,
            min_circularity=0.01,
            min_solidity=0.5
        )

        self.bounding_boxes, self.areas = self.extract_bounding_boxes_from_mask(mask_shape)
        self.classifications = []
        self.avg_colors = []

        print(f"\nFound {len(self.bounding_boxes)} strawberry regions")
        print("Region | Area    | Width x Height | Classification | Avg RGB | Avg HSV | Avg LAB")
        print("-" * 90)

        for i, (bbox, area) in enumerate(zip(self.bounding_boxes, self.areas)):
            avg_colors = self.calculate_average_color_in_bbox(img_rgb, bbox)
            classification = self.classify_strawberry(avg_colors)
            
            self.avg_colors.append(avg_colors)
            self.classifications.append(classification)
            
            if avg_colors:
                rgb_str = f"({avg_colors['rgb'][0]:.0f},{avg_colors['rgb'][1]:.0f},{avg_colors['rgb'][2]:.0f})"
                hsv_str = f"({avg_colors['hsv'][0]:.0f},{avg_colors['hsv'][1]:.2f},{avg_colors['hsv'][2]:.2f})"
                lab_str = f"({avg_colors['lab'][0]:.0f},{avg_colors['lab'][1]:.0f},{avg_colors['lab'][2]:.0f})"
                size_str = f"{bbox['width']}x{bbox['height']}"
                print(f"{i+1:6d} | {area:7.0f} | {size_str:12s} | {classification:12s} | {rgb_str:12s} | {hsv_str:12s} | {lab_str}")

        # Count classifications
        classification_counts = {}
        for cls in self.classifications:
            classification_counts[cls] = classification_counts.get(cls, 0) + 1

        
        
        print(f"\nClassification Summary:")
        for cls, count in classification_counts.items():
            print(f"  {cls}: {count}")

        # Plotting
        self.ax_orig.clear()
        self.ax_orig.imshow(img_rgb)
        self.ax_orig.set_title('Original Image')
        self.ax_orig.axis('off')

        total_valid = sum(cnt for lbl, cnt in classification_counts.items() if lbl != 'Unknown')

        self.ax_mask.clear()
        legend_handles = []
        for cls, count in classification_counts.items():
            color = np.array(self.get_classification_color(cls)) / 255.0
            legend_handles.append(
                Patch(edgecolor=color, facecolor='none',
                    label=f"{cls.capitalize()}: {count}"))
        self.ax_mask.legend(handles=legend_handles, loc='center', frameon=False, title="Classification Legend")
        self.ax_mask.legend(handles=legend_handles, loc='center', frameon=False, title=f"Number of strawberries detected: {total_valid}")
        self.ax_mask.axis('off')
        

        # Plot segmented result
        self.ax_res.clear()
        segmented = img_rgb.copy()
        segmented[~mask_shape] = 0
        self.ax_res.imshow(segmented)
        self.ax_res.set_title('Segmented Strawberries')
        self.ax_res.axis('off')

        # Plot classification result with colored bounding boxes
        self.ax_class.clear()
        self.ax_class.imshow(img_rgb)
        
        for bbox, cls in zip(self.bounding_boxes, self.classifications):
            color = np.array(self.get_classification_color(cls)) / 255.0
            rect = Rectangle((bbox['minc'], bbox['minr']),
                         bbox['width'], bbox['height'],
                         fill=False, edgecolor=color, linewidth=2)
            self.ax_class.add_patch(rect)

        self.ax_class.set_title('Classification with Bounding Boxes')
        self.ax_class.axis('off')


        plt.tight_layout()
        plt.draw()

    def show(self):
        plt.show()

    def export_results(self, output_path="results"):
        """Export classification results to CSV file."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        current_file = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(current_file))[0]
        csv_file = os.path.join(output_path, f"{base_name}_classification.csv")
        # Fine Tuning Parameters
        with open(csv_file, 'w') as f:
            f.write("Region,Area,Width,Height,Classification,Avg_R,Avg_G,Avg_B,Avg_H,Avg_S,Avg_V,Avg_L,Avg_A,Avg_B_lab,MinR,MinC,MaxR,MaxC\n")
            
            for i, (bbox, area, classification, avg_colors) in enumerate(zip(self.bounding_boxes, self.areas, self.classifications, self.avg_colors)):
                if avg_colors:
                    rgb = avg_colors['rgb']
                    hsv = avg_colors['hsv']
                    lab = avg_colors['lab']
                    f.write(f"{i+1},{area:.0f},{bbox['width']},{bbox['height']},{classification},"
                           f"{rgb[0]:.1f},{rgb[1]:.1f},{rgb[2]:.1f},"
                           f"{hsv[0]:.1f},{hsv[1]:.3f},{hsv[2]:.3f},"
                           f"{lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f},"
                           f"{bbox['minr']},{bbox['minc']},{bbox['maxr']},{bbox['maxc']}\n")
        
        print(f"Results exported to: {csv_file}")

if __name__ == "__main__":
    # Uses only scikit-image for bounding box detection and classification
    
    navigator = StrawberryBoundingBoxClassifier("dataset")
    navigator.show()
    
    # Optionally export results after processing
    # navigator.export_results()