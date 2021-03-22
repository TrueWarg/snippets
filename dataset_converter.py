import json
import os


def convert_coco_like_to_voc(dataset_source_path: str, out_path: str):
    with open(dataset_source_path) as f:
        j = json.load(f)
        origin_images = j['images']
        origin_annotations = j['annotations']

    for image in origin_images:
        file_name = image['file_name']
        image_id = image['id']

        selection = []
        for annotation in origin_annotations:
            if annotation['image_id'] == image_id:
                selection.append(annotation)

        boxes = []
        labels = []
        for s in selection:
            x, y, w, h, angle = s['bbox']
            xmin, ymin = (int(x) - int(w / 2), int(y) - int(h / 2))
            xmax, ymax = (int(x) + int(w / 2), int(y) + int(h / 2))
            box = [xmin, ymin, xmax, ymax, angle]
            boxes.append(box)
            labels.append(s['category_id'])

        image_name_without_extension = os.path.splitext(file_name)[0]
        out_annotation_file_name = os.path.join(out_path, 'annotation', f'{image_name_without_extension}.xml')

        with open(out_annotation_file_name, 'w') as out_file:
            out_file.write("<annotation>\n")
            out_file.write("<folder>VOC2007</folder>\n")
            out_file.write(f"<filename>{out_annotation_file_name}</filename>\n")
            out_file.write(
                "<size> <width>800</width> <height>800</height><depth>3</depth></size><segmented>0</segmented>\n")

            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax, angle = box

                out_file.write("<object>\n")
                out_file.write(f"<name>{label}</name>\n")
                out_file.write(f"<name>Rear</name>\n")
                out_file.write(f"<truncated>0</truncated>\n")
                out_file.write(f"<difficult>0</difficult>\n")
                out_file.write(f"<bndbox>\n")
                out_file.write(f"  <xmin>{xmin}</xmin>\n")
                out_file.write(f"  <ymin>{ymin}</ymin>\n")
                out_file.write(f"  <xmax>{xmax}</xmax>\n")
                out_file.write(f"  <ymax>{ymax}</ymax>\n")
                out_file.write(f"  <angle>{angle}</angle>\n")
                out_file.write(f"</bndbox>\n")
                out_file.write("</object>\n")

            out_file.write("</annotation>\n")

    image_names_without_extension = map(lambda x: os.path.splitext(x)[0], origin_images)
    with open(os.path.join(out_path, 'trainval.txt'), 'w') as out_file:
        out_file.write("\n".join(image_names_without_extension))


if __name__ == '__main__':
    convert_coco_like_to_voc(dataset_source_path='', out_path='')
