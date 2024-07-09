clc;
clear;

f = input('Enter the filename of the grayscale image (e.g., Penguins_grey.png): ', 's');
g = input('Enter the filename of the RGB image (e.g., Penguins_RGB.png): ', 's');
F = imread(f);
G = imread(g);

disp('Class of F:'); disp(class(F));
disp('Class of G:'); disp(class(G));

disp('Size of F:'); disp(size(F,1:2));
disp('Size of G:'); disp(size(G));

choice = menu('Choose A Function To Execute:', ...
              '1. Check Number Of Input Arguments PART 1', ...
              '2. Check Dimensions Of Input Arguments PART 2', ...
              '3. Get Image Info PART 3', ...
              '4. Shift Center To Corners PART 4', ...
              '5. Blurred (Scale) Image PART 5', ...
              '6. Resize Image PART 5', ...
              '7. Resizing Image And Putting It On Original Image PART 6', ...
              '8. Shifted Image PART 7', ...
              '9. Cut Piece of image And then put it back PART 8', ...
              '10. Brightness PART 9', ...
              '11. Adding Noise(by my own choice)PART 10', ...
              '12. FFT PART 11', ...
              '13. Applying  Math Function To Picture PART 12', ...
              '14. Histogram PART 13', ...
              '15. (by my own choice) Changing The Grey Level PART 14', ...
              '16. Print Text On Picture PART 15', ...
              '17. (by my own choice) Making Frame For Picture Part 16');
switch choice
    case 1
        checking_number_of_inputarguments(F);
        checking_number_of_inputarguments(G);
    case 2
        checking_dimensions_of_inputarguments(F, true);
        checking_dimensions_of_inputarguments(G, false);

    case 3
        F_info = get_image_info(F);
        G_info = get_image_info(G);
        disp('F Info:'); disp(F_info);
        disp('G Info:'); disp(G_info);
    case 4
        F_shifted = shift_center_to_corners(F);
        G_shifted = shift_center_to_corners(G);
    case 5
        k = input('Enter a scalar value (1-100) for scaling: ');
        if k < 1 || k > 100
            error('Scalar value must be between 1 and 100.');
        end
        F_scaled = scale_image(F, k);
        G_scaled = scale_image(G, k);
    case 6
        K = input('Enter the scale percentage (1 to 100): ');
        grey_image_filename = f;
        rgb_image_filename = g;
        resized_grey_image = resize_and_save_image(grey_image_filename, K);
        resized_rgb_image = resize_and_save_image(rgb_image_filename, K);
        figure;
        imshow(resized_grey_image);
        title('Grey Image');
        figure;
        imshow(resized_rgb_image);
        title('RGB Image');


    case 7
        S = input('Enter the scale percentage (1 to 100): ');
        grey_image_filename_overlay_resized_image = f;
        rgb_image_filename_overlay_resized_image = g;
        overlaid_image = overlay_resized_image(F, G, S);

    case 8 

        p = input('Enter the shift value for x (horizontal shift): ');
        t = input('Enter the shift value for y (vertical shift): ');
        grey_image_filename = f;
        rgb_image_filename = g;
        shifted_grey_image = rotate_and_shift_image(grey_image_filename, p, t);
        shifted_rgb_image = rotate_and_shift_image(rgb_image_filename, p, t);
        figure;
        imshow(shifted_grey_image);
        title('Shifted Grey Image');
        figure;
        imshow(shifted_rgb_image);
        title('Shifted RGB Image');
     

    case 9
        x = input('Enter the x coordinate (top-left corner) for cropping: ');
        y = input('Enter the y coordinate (top-left corner) for cropping: ');
        w = input('Enter the width of the cropping rectangle: ');
        u = input('Enter the height of the cropping rectangle: ');
        grey_image_filename = f;
        rgb_image_filename = g;
        [processed_grey_image, cropped_grey_image] = crop_and_replace_image(grey_image_filename, x, y, w, u);
        [processed_rgb_image, cropped_rgb_image] = crop_and_replace_image(rgb_image_filename, x, y, w, u);
        figure;
        imshow(cropped_grey_image);
        title('Cropped Grey Image');
        figure;
        imshow(processed_grey_image);
        title('Processed Grey Image');
        figure;
        imshow(processed_rgb_image);
        title('Processed RGB Image');
    case 10
        k = input('Enter the scaling percentage (e.g., 50 for 50%): ');
        grey_image_filename = f;
        rgb_image_filename = g;
        scaled_grey_image = scale_image_mean(grey_image_filename, k);
        scaled_rgb_image = scale_image_mean(rgb_image_filename, k);
        figure;
        imshow(scaled_grey_image);
        title('Scaled Grey Image');
        figure;
        imshow(scaled_rgb_image);
        title('Scaled RGB Image');

    case 11
        applyNoise('Penguins_grey.png', 'Penguins_RGB.png');

    case 12
        plot_2D_FFT(f);
        plot_2D_FFT(g);

    case 13
        expression = input('Enter expression (e.g., "sin(x) + cos(y)"): ', 's');
        if ~isstr(expression)
            error('Invalid input. Please enter a string.');
        end
        result_image1_F = applyExpressionToImage(F, expression);
        result_image2_F = applyExpressionToImage(F, expression);
        result_image1_G = applyExpressionToImage(G, expression);
        result_image2_G = applyExpressionToImage(G, expression);

        subplot(2, 2, 1);
        imshow(result_image1_F);
        title(['Result of ' expression ' on grey image']);
        subplot(2, 2, 2);
        imshow(result_image2_F);
        title(['Result of ' expression ' on grey image']);
        subplot(2, 2, 3);
        imshow(result_image1_G);
        title(['Result of ' expression ' on RGB image']);
        subplot(2, 2, 4);
        imshow(result_image2_G);
        title(['Result of ' expression ' on RGB image']);

    case 14
        plotImageHistograms(F, G);

    case 15
        processImage(f);
        processImage(g);

    case 16
        applyTextToImages({f, g});

    case 17
        F = {f, g};
        framed_images = create_frame(F);

end


%PART1

function checking_number_of_inputarguments(varargin)
    if nargin < 1
        error('At least one input argument is required.');
    end
    disp('Input arguments are valid.');
    if nargin >= 1
        image = varargin{1};
    end
    if nargin > 1
    end
end

%PART2

function checking_dimensions_of_inputarguments(image, is_grayscale)
    if nargin < 2
        error('At least two input arguments are required.');
    end
    
    if ~isnumeric(image)
        error('The input image must be a numeric array.');
    end
    
    disp('Input image is valid.');
    
    dims = size(image);
    
    if is_grayscale
        fprintf('Grayscale image dimensions: %d x %d\n', dims(1), dims(2));
    else
        if ndims(image) == 3 && size(image, 3) == 3
            fprintf('RGB image dimensions: %d x %d x 3\n', dims(1), dims(2));
        else
            error('The image array should have three color channels for RGB images.');
        end
    end
    
end

%PART3

function image_info = get_image_info(image)
    if ~isnumeric(image)
        error('The input argument must be a numeric array.');
    end
    [m, n, c] = size(image);
    if n == 493
        image_type = 'Grayscale';
    elseif c == 3
        image_type = 'RGB';
    else
        error('Unsupported image type.');
    end
    if strcmp(image_type, 'RGB')
        R = image(:,:,1);
        G = image(:,:,2);
        B = image(:,:,3);
    else
        R = [];
        G = [];
        B = [];
    end
    image_info = struct('Size', [m, n], 'Type', image_type, 'R', R, 'G', G, 'B', B);
    if strcmp(image_type, 'RGB')
        figure;
        subplot(2, 2, 1); imshow(image); title('Original Image');
        subplot(2, 2, 2); imshow(R); title('Red Channel');
        subplot(2, 2, 3); imshow(G); title('Green Channel');
        subplot(2, 2, 4); imshow(B); title('Blue Channel');
    end
end

%PART4

function shifted_image = shift_center_to_corners(image)
    if ~isnumeric(image)
        error('The input argument must be a numeric array.');
    end
    [m, n, c] = size(image);
    center_x = round(n / 2);
    center_y = round(m / 2);
    corners = [1, 1; 1, m; n, 1; n, m];
    shifted_image = zeros(m, n, c, 4, 'like', image);
    for i = 1:4
        corner_x = corners(i, 1);
        corner_y = corners(i, 2);
        tx = corner_x - center_x;
        ty = corner_y - center_y;
        shifted_image(:,:,:,i) = circshift(image, [ty, tx]);
    end
    for i = 4
        corner_x = corners(i, 1);
        corner_y = corners(i, 2);
        tx = corner_x - center_x;
        ty = corner_y - center_y;
        shifted_image = circshift(image, [ty, tx]);
    end
    figure;
    subplot(1, 2, 1); imshow(image); title('Original Image');
    subplot(1, 2, 2); imshow(shifted_image); title('Shifted to Fourth Corner');
end

%PART5

function scaled_image = scale_image(image, k)
    scale_factor = k / 100;
    [m, n, c] = size(image);
    new_m = round(m * scale_factor);
    new_n = round(n * scale_factor);
    scaled_image = imresize(image, [new_m, new_n], 'nearest'); 
    figure;
    subplot(1, 2, 1); imshow(image); title('Original Image');
    subplot(1, 2, 2); imshow(scaled_image); title(['Scaled Image by ', num2str(k), '%']);
end

%PART5

function resized_image = resize_and_save_image(image_filename, K)
    image = imread(image_filename);
    [rows, cols, ~] = size(image);
    scale = K / 100;
    new_rows = round(rows * scale);
    new_cols = round(cols * scale);
    resized_image = imresize(image, [new_rows, new_cols]);
    [pathstr, name, ext] = fileparts(image_filename);
    output_filename = fullfile(pathstr, [name, '_resized_', num2str(K), 'percent', ext]);
    imwrite(resized_image, output_filename);
    disp(['Resized image saved as ', output_filename]);
end


%PART6
function overlaid_image = overlay_resized_image(original_image, resized_image, S)
    [rows_original, cols_original, ~] = size(original_image);
    scale = S / 100;
    new_rows = round(rows_original * scale);
    new_cols = round(cols_original * scale);
    resized_image = imresize(resized_image, [new_rows, new_cols]);
    [rows_resized, cols_resized, ~] = size(resized_image);
    row_start = floor((rows_original - rows_resized) / 2) + 1;
    col_start = floor((cols_original - cols_resized) / 2) + 1;
    overlaid_image = original_image;
    overlaid_image(row_start:row_start+rows_resized-1, col_start:col_start+cols_resized-1, :) = resized_image;
    imshow(overlaid_image);
    imwrite(overlaid_image, 'overlaid_image.png');
    disp('Overlaid image saved as "overlaid_image.png"');
end


%PART7

function shifted_image = rotate_and_shift_image(image_filename, p, t)
    image = imread(image_filename);
    [rows, cols, ~] = size(image);
    shifted_image = uint8(zeros(rows, cols, size(image, 3)));
    rotation_matrix = [1 0 0; 0 1 0; p t 1];
    tform = affine2d(rotation_matrix);
    shifted_image = imwarp(image, tform, 'OutputView', imref2d(size(image)));
    [pathstr, name, ext] = fileparts(image_filename);
    output_filename = fullfile(pathstr, [name, '_shifted_', num2str(p), '_', num2str(t), ext]);
    imwrite(shifted_image, output_filename);
    disp(['Shifted image saved as ', output_filename]);
end


%Part8
function [processed_image, cropped_image] = crop_and_replace_image(image_filename, x, y, w, u)
    image = imread(image_filename);
    [rows, cols, num_channels] = size(image);
    x = max(1, min(x, cols));
    y = max(1, min(y, rows));
    w = max(1, min(w, cols - x + 1));
    u = max(1, min(u, rows - y + 1));
    cropped_image = image(y:y+u-1, x:x+w-1, :);
    imshow(cropped_image);
    [pathstr, name, ext] = fileparts(image_filename);
    cropped_output_filename = fullfile(pathstr, [name, '_cropped_', num2str(x), '_', num2str(y), '_', num2str(w), '_', num2str(u), ext]);
    imwrite(cropped_image, cropped_output_filename);
    disp(['Cropped image saved as ', cropped_output_filename]);
    processed_image = image;
    processed_image(y:y+u-1, x:x+w-1, :) = cropped_image;
    processed_output_filename = fullfile(pathstr, [name, '_processed_', num2str(x), '_', num2str(y), '_', num2str(w), '_', num2str(u), ext]);
    imwrite(processed_image, processed_output_filename);
    disp(['Processed image saved as ', processed_output_filename]);
end




%Part9

function scaled_image = scale_image_mean(image_filename, k)
    image = imread(image_filename);
    if size(image, 3) == 1
        mean_val = mean(image(:));
        scaled_mean_val = mean_val * (k / 100);
        scaled_image = double(image) * (scaled_mean_val / mean_val);
        scaled_image = uint8(min(max(scaled_image, 0), 255));
    else
        mean_R = mean(image(:,:,1), 'all');
        mean_G = mean(image(:,:,2), 'all');
        mean_B = mean(image(:,:,3), 'all');
        scaled_mean_R = mean_R * (k / 100);
        scaled_mean_G = mean_G * (k / 100);
        scaled_mean_B = mean_B * (k / 100);
        scaled_image = image;
        scaled_image(:,:,1) = double(image(:,:,1)) * (scaled_mean_R / mean_R);
        scaled_image(:,:,2) = double(image(:,:,2)) * (scaled_mean_G / mean_G);
        scaled_image(:,:,3) = double(image(:,:,3)) * (scaled_mean_B / mean_B);
        scaled_image(:,:,1) = uint8(min(max(scaled_image(:,:,1), 0), 255));
        scaled_image(:,:,2) = uint8(min(max(scaled_image(:,:,2), 0), 255));
        scaled_image(:,:,3) = uint8(min(max(scaled_image(:,:,3), 0), 255));
    end
    [pathstr, name, ext] = fileparts(image_filename);
    output_filename = fullfile(pathstr, [name, '_scaled_', num2str(k), 'percent', ext]);
    imwrite(scaled_image, output_filename);
    disp(['Scaled image saved as ', output_filename]);
end



%PART10


function applyNoise(grayFilename, rgbFilename)
    F = imread(grayFilename);
    F_gaussian_noisy = imnoise(F, 'gaussian', 0, 0.01);
    F_salt_pepper_noisy = imnoise(F, 'salt & pepper', 0.05); 
    G = imread(rgbFilename);
    G_gaussian_noisy = imnoise(G, 'gaussian', 0, 0.01); 
    G_salt_pepper_noisy = imnoise(G, 'salt & pepper', 0.05);
    figure;
    subplot(3, 2, 1);
    imshow(F);
    title('Original Grayscale Image');

    subplot(3, 2, 2);
    imshow(G);
    title('Original RGB Image');

    subplot(3, 2, 3);
    imshow(F_gaussian_noisy);
    title('Gaussian Noisy Grayscale Image');

    subplot(3, 2, 4);
    imshow(G_gaussian_noisy);
    title('Gaussian Noisy RGB Image');

    subplot(3, 2, 5);
    imshow(F_salt_pepper_noisy);
    title('Salt & Pepper Noisy Grayscale Image');

    subplot(3, 2, 6);
    imshow(G_salt_pepper_noisy);
    title('Salt & Pepper Noisy RGB Image');
end


%PART11

function plot_2D_FFT(image_filename)
    image = imread(image_filename);
    image_double = double(image);
    fft_image = fft2(image_double);
    fft_image_shifted = fftshift(fft_image);
    magnitude_spectrum = abs(fft_image_shifted);
    phase_spectrum = angle(fft_image_shifted);
    figure; 
    subplot(3, 3, 1);
    imshow(image);
    title('Original Image');
    subplot(3, 3, 2);
    imshow(log(1 + magnitude_spectrum), []);
    title('Magnitude Spectrum');
    subplot(3, 3, 3);
    imshow(phase_spectrum, []);
    title('Phase Spectrum');
    reconstructed_image = ifft2(ifftshift(fft_image_shifted));
    subplot(3, 3, 4);
    imshow(uint8(reconstructed_image));
    title('Reconstructed Image');
    imdata = imread(image_filename);
    if size(imdata, 3) == 3
        imdata = rgb2gray(imdata);
    end
    F = fft2(imdata);
    S = abs(F);
    subplot(3, 3, 5);
    imshow(S, []);
    title('Fourier transform of Second Image');
    Fsh = fftshift(F);
    subplot(3, 3, 6);
    imshow(abs(Fsh), []);
    title('Centered Fourier transform of Second Image');
    S2 = log(1 + abs(Fsh));
    subplot(3, 3, 7);
    imshow(S2, []);
    title('Log Transformed Second Image');
    F = ifftshift(Fsh);
    f = ifft2(F);
    subplot(3, 3, 8);
    imshow(f, []);
    title('Reconstructed Second Image');
end

%part12
function result_image = applyExpressionToImage(image, expression)
    [rows, cols, ~] = size(image);
    x = linspace(0, pi/4, cols);
    y = linspace(0, pi/4, rows);
    [X, Y] = meshgrid(x, y);
    expression = strrep(expression, 'x', 'X');
    expression = strrep(expression, 'y', 'Y');
    result_image = eval(expression);
end




%part13
function plotImageHistograms(F, G)
    figure;
    imhist(F);
    title('Histogram of Grayscale Image');
    figure;
    subplot(3,1,1);
    imhist(G(:,:,1));
    title('Histogram of Red Channel (RGB Image)');
    subplot(3,1,2);
    imhist(G(:,:,2));
    title('Histogram of Green Channel (RGB Image)');
    subplot(3,1,3);
    imhist(G(:,:,3));
    title('Histogram of Blue Channel (RGB Image)');
end

%PART14


function processImage(filename)
    F = imread(filename);
    F_brighter = F + 50;
    F_brighter(F_brighter > 255) = 255;
    F_equalized = histeq(F);
    figure;
    subplot(1, 3, 1);
    imshow(F);
    title('Original Grayscale Image');

    subplot(1, 3, 2);
    imshow(F_brighter);
    title('Brighter Image');

    subplot(1, 3, 3);
    imshow(F_equalized);
    title('Histogram Equalized Image');
end

%PART15
function applyTextToImages(image_files)
    text = input('Enter the text you want to overlay on the images: ', 's');
    option = input('Choose the scaling option (2x, 4x, 10x): ', 's');
    scale = input('Enter the text size (15): ');
    for i = 1:numel(image_files)
        img = imread(image_files{i});
        [rows, cols, ~] = size(img);
        center_x = round(cols / 2);
        center_y = round(rows / 2);
        switch option
            case '2x'
                text_scale = scale * 2;
            case '4x'
                text_scale = scale * 4;
            case '10x'
                text_scale = scale * 10;
            otherwise
                error('Invalid option.');
        end
        text_img = insertText(zeros(rows, cols), [center_x center_y], text, ...
                              'FontSize', text_scale, 'BoxColor', 'black', 'TextColor', 'white', ...
                              'AnchorPoint', 'Center');
        figure;
        imshow(imfuse(img, text_img, 'blend'));
    end
end

%Part16

function framed_images = create_frame(images)
    framed_images = cell(1, numel(images));
    frame_size = 10;
    for i = 1:numel(images)
        img = imread(images{i});
        [rows, cols, ~] = size(img);
        frame = uint8(ones(rows + 2 * frame_size, cols + 2 * frame_size, 3) * 255);
        frame(frame_size + 1:frame_size + rows, frame_size + 1:frame_size + cols, :) = img;
        frame(1:frame_size, :, :) = 0;
        frame(end - frame_size + 1:end, :, :) = 0;
        frame(:, 1:frame_size, :) = 0;
        frame(:, end - frame_size + 1:end, :) = 0;
        framed_images{i} = frame;
    end
    for i = 1:numel(framed_images)
        figure;
        imshow(framed_images{i});
    end
end


