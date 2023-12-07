import os
import re
import numpy as np
import pdf2image
import imageio
import video_render_ffmpeg

from util.video_render.video_renderer_data import validation_losses

from util.video_render.video_renderer_data import training_losses

#####################
# PARAMS:
# p_epoch_count = 32
p_epoch_count = 275
p_epoch_offset = 5
p_input_dir = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\production\\paper_candidate_2\\metrics_live\\sigmoid_live\\naive\\ESM36'
p_output_dir = 'Z:\\nilfoe\\Python\\omnisphero-mil\\util\\video_render\\out'
p_filename_regex = 'ESM36-prediction_map-epoch(\\d+).csv'


def render_frames(epoch_count: int, epoch_offset: int, input_dir: str, output_dir):
    epoch_file_count = epoch_count / epoch_offset
    print('Input dir: ' + input_dir)

    f = open('frame_template.txt', 'r')
    frame_template = f.read()
    f.close()
    rendered_image_file_names = []
    rendered_images = []

    for i in range(epoch_count):
        # Start. Printing progress.
        print(str(i + 1) + '/' + str(epoch_count))

        file_name_prediction_map = 'ESM36-prediction_map-epoch' + str(i + 1) + '.csv'
        file_name_sigmoid_fit = 'ESM36-sigmoid_fit-epoch' + str(i + 1) + '.csv'
        file_path_prediction_map = input_dir + os.sep + file_name_prediction_map
        file_path_sigmoid_fit = input_dir + os.sep + file_name_sigmoid_fit

        if not os.path.exists(file_path_prediction_map) or not os.path.exists(file_path_sigmoid_fit):
            continue

        f = open(file_path_prediction_map, 'r')
        prediction_map = f.readlines()
        f.close()

        f = open(file_path_sigmoid_fit, 'r')
        sigmoid_fit = f.readlines()
        f.close()

        avg_predictions = extract_prediction_map(prediction_map)
        fit_graph = extract_sigmoid_fit(sigmoid_fit)

        prediction_graph = []
        for j in range(len(avg_predictions)):
            prediction_graph.append((float(j + 2), avg_predictions[j]))

        prediction_graph_text = '\n' + '\n'.join([str(g) for g in prediction_graph]) + '\n'
        fit_graph_text = '\n' + '\n'.join([str(g) for g in fit_graph]) + '\n'
        training_loss_text = get_training_loss_text(i + 1)
        validation_loss_text = get_validation_loss_text(i + 1)

        # writing the .tex file!
        out_file_name = output_dir + os.sep + 'frame-' + str(i) + '.tex'
        out_text = frame_template

        out_text = out_text.replace('<#validation_loss#>', validation_loss_text)
        out_text = out_text.replace('<#training_loss#>', training_loss_text)
        out_text = out_text.replace('<#curve_fit#>', fit_graph_text)
        out_text = out_text.replace('<#mean_predictions#>', prediction_graph_text)
        out_text = out_text.replace('<#epoch#>', str(i + 1))

        f = open(out_file_name, 'w')
        f.write(out_text)
        f.close()

        # Done. Now compiling it!
        command = 'pdflatex ' + out_file_name + ' -output-directory ' + output_dir
        print('Compile command: ' + command)
        os.system(command)

        im_render_out_name = out_file_name[:-3] + 'png'
        pages = pdf2image.convert_from_path(out_file_name[:-3] + 'pdf', 800)
        for page in pages:
            page.save(im_render_out_name, 'PNG')

        rendered_image_file_names.append(im_render_out_name)
        rendered_images.append(imageio.imread(im_render_out_name))
        # Done, next

    print('Rendering the .gif now')
    composite_out_dir = output_dir + os.sep + 'comp'
    os.makedirs(composite_out_dir, exist_ok=True)
    imageio.mimsave(composite_out_dir + os.sep + 'composite.gif', rendered_images)

    f = open(composite_out_dir + os.sep + 'frames.txt', 'w')
    for frame in rendered_image_file_names:
        f.write(frame + '\n')
    f.close()

    # Rendering avi
    print('Rendering video.')
    video_render_ffmpeg.render_images_to_video_multiple(rendered_image_file_names, composite_out_dir + os.sep, fps=0.5)

    print('All done.')


def get_training_loss_text(index):
    return '\n' + '\n'.join([str(g) for g in training_losses[0:index]]) + '\n'


def get_validation_loss_text(index):
    return '\n' + '\n'.join([str(g) for g in validation_losses[0:index]]) + '\n'


def extract_prediction_map(prediction_map):
    averages = []

    for i in range(1, 8):
        predictions = []
        for j in range(1, 6):
            line = prediction_map[j]
            line = line.split(';')
            cell = str(line[i]).strip()
            if len(cell) > 0:
                predictions.append(float(cell))

        avg = np.average(predictions)
        averages.append(avg)

    return averages


def extract_sigmoid_fit(sigmoid_fit):
    fit_graph = []

    for l in sigmoid_fit[1:]:
        l = l.split(';')
        fit_graph.append((float(l[1].strip()), float(l[2].strip())))

    return fit_graph


def main():
    render_frames(epoch_count=p_epoch_count,
                  input_dir=p_input_dir,
                  epoch_offset=p_epoch_offset,
                  output_dir=p_output_dir)


if __name__ == '__main__':
    main()
