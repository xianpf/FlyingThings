from src.utility.SetupUtility import SetupUtility
SetupUtility.setup_pip(["scikit-image"])

import datetime
from itertools import groupby
import csv
import json
import os
import shutil
import numpy as np
from skimage import measure

import bpy

from src.utility.Utility import Utility

class CocoWriterUtility:

    @staticmethod
    def write(output_dir: str, mask_encoding_format="rle", supercategory="coco_annotations", append_to_existing_output=False,
                segmap_output_key="segmap", segcolormap_output_key="segcolormap", rgb_output_key="colors"):
        """ Writes coco annotations in the following steps:
        1. Locate the seg images
        2. Locate the rgb maps
        3. Locate the seg mappings
        4. Read color mappings
        5. For each frame write the coco annotation

        :param output_dir: Output directory to write the coco annotations
        :param mask_encoding_format: Encoding format of the binary masks. Default: 'rle'. Available: 'rle', 'polygon'.
        :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset set by 'bop_dataset_name' or 
            any loaded object with specified 'cp_supercategory'
        :param append_to_existing_output: If true and if there is already a coco_annotations.json file in the output directory, the new coco
            annotations will be appended to the existing file. Also the rgb images will be named such that there are
            no collisions. Default: False.
        :param segmap_output_key: The output key with which the segmentation images were registered. Should be the same as the output_key
            of the SegMapRenderer module. Default: segmap.
        :param segcolormap_output_key: The output key with which the csv file for object name/class correspondences was registered. Should be
            the same as the colormap_output_key of the SegMapRenderer module. Default: segcolormap.
        :param rgb_output_key: The output key with which the rgb images were registered. Should be the same as the output_key of the
            RgbRenderer module. Default: colors.
        """

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Find path pattern of segmentation images
        segmentation_map_output = Utility.find_registered_output_by_key(segmap_output_key)
        if segmentation_map_output is None:
            raise Exception("There is no output registered with key {}. Are you sure you ran the SegMapRenderer module "
                            "before?".format(segmap_output_key))
        
        # Find path pattern of rgb images
        rgb_output = Utility.find_registered_output_by_key(rgb_output_key)
        if rgb_output is None:
            raise Exception("There is no output registered with key {}. Are you sure you ran the RgbRenderer module "
                            "before?".format(rgb_output_key))
    
        # collect all segmaps
        segmentation_map_paths = []

        # Find path of name class mapping csv file
        segcolormap_output = Utility.find_registered_output_by_key(segcolormap_output_key)
        if segcolormap_output is None:
            raise Exception("There is no output registered with key {}. Are you sure you ran the SegMapRenderer module "
                            "with 'map_by' set to 'instance' before?".format(segcolormap_output_key))

        # read colormappings, which include object name/class to integer mapping
        inst_attribute_maps = []
        with open(segcolormap_output["path"], 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for mapping in reader:
                inst_attribute_maps.append(mapping)

        coco_annotations_path = os.path.join(output_dir, "coco_annotations.json")
        # Calculate image numbering offset, if append_to_existing_output is activated and coco data exists
        if append_to_existing_output and os.path.exists(coco_annotations_path):
            with open(coco_annotations_path, 'r') as fp:
                existing_coco_annotations = json.load(fp)
            image_offset = max([image["id"] for image in existing_coco_annotations["images"]]) + 1
        else:
            image_offset = 0
            existing_coco_annotations = None

        # collect all RGB paths
        new_coco_image_paths = []
        # for each rendered frame
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            segmentation_map_paths.append(segmentation_map_output["path"] % frame)

            source_path = rgb_output["path"] % frame
            target_path = os.path.join(output_dir, os.path.basename(rgb_output["path"] % (frame + image_offset)))

            shutil.copyfile(source_path, target_path)
            new_coco_image_paths.append(os.path.basename(target_path))

        coco_output = CocoWriterUtility.generate_coco_annotations(segmentation_map_paths,
                                                            new_coco_image_paths,
                                                            inst_attribute_maps,
                                                            supercategory,
                                                            mask_encoding_format,
                                                            existing_coco_annotations)

        print("Writing coco annotations to " + coco_annotations_path)
        with open(coco_annotations_path, 'w') as fp:
            json.dump(coco_output, fp)

    @staticmethod
    def generate_coco_annotations(segmentation_map_paths, image_paths, inst_attribute_maps, supercategory,
                                  mask_encoding_format, existing_coco_annotations=None):
        """Generates coco annotations for images

        :param segmentation_map_paths: A list of paths which points to the rendered segmentation maps.
        :param image_paths: A list of paths which points to the rendered segmentation maps.
        :param inst_attribute_maps: mapping with idx, class and optionally supercategory/bop_dataset_name
        :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset
        :param mask_encoding_format: Encoding format of the binary mask. Type: string.
        :param existing_coco_annotations: If given, the new coco annotations will be appended to the given coco annotations dict.
        :return: dict containing coco annotations
        """

        categories = []
        instance_2_category_map = {}

        for inst in inst_attribute_maps:
            # skip background
            if int(inst["category_id"]) != 0:
                # take all objects or objects from specified supercategory is defined
                inst_supercategory = "coco_annotations"
                if "bop_dataset_name" in inst:
                    inst_supercategory = inst["bop_dataset_name"]
                elif "supercategory" in inst:
                    inst_supercategory = inst["supercategory"]

                if supercategory == inst_supercategory or supercategory == 'coco_annotations':
                    cat_dict = {'id': int(inst["category_id"]),
                                'name': inst["category_id"],
                                'supercategory': inst_supercategory}
                    if cat_dict not in categories:
                        categories.append(cat_dict)
                    instance_2_category_map[int(inst["idx"])] = int(inst["category_id"])

        licenses = [{
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }]
        info = {
            "description": supercategory,
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "Unknown",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        images = []
        annotations = []

        for segmentation_map_path, image_path in zip(segmentation_map_paths, image_paths):
            
            # Load instance map
            inst_channel = int(inst_attribute_maps[0]['channel_instance'])
            segmentation_map = np.load(segmentation_map_path)[:,:,inst_channel]

            # Add coco info for image
            image_id = len(images)
            images.append(CocoWriterUtility.create_image_info(image_id, image_path, segmentation_map.shape))

            # Go through all objects visible in this image
            instances = np.unique(segmentation_map)
            # Remove background
            instances = np.delete(instances, np.where(instances == 0))
            for inst in instances:
                if inst in instance_2_category_map:
                    # Calc object mask
                    binary_inst_mask = np.where(segmentation_map == inst, 1, 0)
                    # Add coco info for object in this image
                    annotation = CocoWriterUtility.create_annotation_info(len(annotations),
                                                                    image_id,
                                                                    instance_2_category_map[inst],
                                                                    binary_inst_mask,
                                                                    mask_encoding_format)
                    if annotation is not None:
                        annotations.append(annotation)

        new_coco_annotations = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        if existing_coco_annotations is not None:
            new_coco_annotations = CocoWriterUtility.merge_coco_annotations(existing_coco_annotations, new_coco_annotations)

        return new_coco_annotations

    @staticmethod
    def merge_coco_annotations(existing_coco_annotations, new_coco_annotations):
        """ Merges the two given coco annotation dicts into one.

        Currently this requires both coco annotations to have the exact same categories/objects.
        The "images" and "annotations" sections are concatenated and respective ids are adjusted.

        :param existing_coco_annotations: A dict describing the first coco annotations.
        :param new_coco_annotations: A dict describing the second coco annotations.
        :return: A dict containing the merged coco annotations.
        """

        # Concatenate category sections
        for cat_dict in new_coco_annotations["categories"]:
            if cat_dict not in existing_coco_annotations["categories"]:
                existing_coco_annotations["categories"].append(cat_dict)

        # Concatenate images sections
        image_id_offset = max([image["id"] for image in existing_coco_annotations["images"]]) + 1
        for image in new_coco_annotations["images"]:
            image["id"] += image_id_offset
        existing_coco_annotations["images"].extend(new_coco_annotations["images"])

        # Concatenate annotations sections
        annotation_id_offset = max([annotation["id"] for annotation in existing_coco_annotations["annotations"]]) + 1
        for annotation in new_coco_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
        existing_coco_annotations["annotations"].extend(new_coco_annotations["annotations"])

        return existing_coco_annotations

    @staticmethod
    def create_image_info(image_id, file_name, image_size):
        """Creates image info section of coco annotation

        :param image_id: integer to uniquly identify image
        :param file_name: filename for image
        :param image_size: The size of the image, given as [W, H]
        """
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }

        return image_info

    @staticmethod
    def create_annotation_info(annotation_id, image_id, category_id, binary_mask, mask_encoding_format, tolerance=2):
        """Creates info section of coco annotation

        :param annotation_id: integer to uniquly identify the annotation
        :param image_id: integer to uniquly identify image
        :param category_id: Id of the category
        :param binary_mask: A binary image mask of the object with the shape [H, W].
        :param mask_encoding_format: Encoding format of the mask. Type: string.
        :param tolerance: The tolerance for fitting polygons to the objects mask.
        """

        area = CocoWriterUtility.calc_binary_mask_area(binary_mask)
        if area < 1:
            return None

        bounding_box = CocoWriterUtility.bbox_from_binary_mask(binary_mask)

        if mask_encoding_format == 'rle':
            segmentation = CocoWriterUtility.binary_mask_to_rle(binary_mask)
        elif mask_encoding_format == 'polygon':
            segmentation = CocoWriterUtility.binary_mask_to_polygon(binary_mask, tolerance)
            if not segmentation:
                return None
        else:
            raise RuntimeError("Unknown encoding format: {}".format(mask_encoding_format))

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0,
            "area": area,
            "bbox": bounding_box,
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }
        return annotation_info

    @staticmethod
    def bbox_from_binary_mask(binary_mask):
        """ Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The bounding box represented as [x, y, width, height]
        """
        # Find all columns and rows that contain 1s
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        # Find the min and max col/row index that contain 1s
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Calc height and width
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        return [int(cmin), int(rmin), int(w), int(h)]

    @staticmethod
    def calc_binary_mask_area(binary_mask):
        """ Returns the area of the given binary mask which is defined as the number of 1s in the mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The computed area
        """
        return binary_mask.sum().tolist()

    @staticmethod
    def close_contour(contour):
        """ Makes sure the given contour is closed.

        :param contour: The contour to close.
        :return: The closed contour.
        """
        # If first != last point => add first point to end of contour to close it
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    @staticmethod
    def binary_mask_to_polygon(binary_mask, tolerance=0):
        """Converts a binary mask to COCO polygon representation

         :param binary_mask: a 2D binary numpy array where '1's represent the object
         :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If
                           tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
        # Reverse padding
        contours = contours - 1
        for contour in contours:
            # Make sure contour is closed
            contour = CocoWriterUtility.close_contour(contour)
            # Approximate contour by polygon
            polygon = measure.approximate_polygon(contour, tolerance)
            # Skip invalid polygons
            if len(polygon) < 3:
                continue
            # Flip xy to yx point representation
            polygon = np.flip(polygon, axis=1)
            # Flatten
            polygon = polygon.ravel()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            polygon[polygon < 0] = 0
            polygons.append(polygon.tolist())

        return polygons

    @staticmethod
    def binary_mask_to_rle(binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))

        return rle
