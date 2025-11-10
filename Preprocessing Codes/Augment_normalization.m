% Set the path to your dataset
datasetPath = 'F:\Shell_Intern\Shell_Project\Data images\Resize_Preprocess\';

% Specify the output directory for preprocessed images
outputPath = 'F:\Shell_Intern\Shell_Project\Data images\Augmented_images\';

% List all subdirectories 
classList = dir(datasetPath);
classList = classList([classList.isdir] & ~ismember({classList.name}, {'.', '..'}));

% Loop over each class
for classIdx = 1:length(classList)
    className = classList(classIdx).name;
    classPath = fullfile(datasetPath, className);

    % Create a subdirectory for the preprocessed images
    outputClassPath = fullfile(outputPath, className);
    if ~exist(outputClassPath, 'dir')
        mkdir(outputClassPath);
    end

    % List all images in the class directory
    imageFiles = dir(fullfile(classPath, '*.jpg'));  % Adjust the file extension as needed

    % Loop over each image in the class
    for imageIdx = 1:length(imageFiles)
        % Read the current image
        imagePath = fullfile(classPath, imageFiles(imageIdx).name);
        currentImage = imread(imagePath);

        % Preprocess the image 
        % Define the target size
        targetSize = [128, 128, 3];
        % Read the image from the imageDatastore
      
        % Resize the image
        resizedImage = imresize(currentImage, targetSize(1:2));
        % Perform zero-mean normalization
        normalizedImage = (double(resizedImage) - mean(double(resizedImage(:)))) / std(double(resizedImage(:)));

        % Store the preprocessed image in the new cell array
        preprocessedImages{imageIdx} = double(normalizedImage);
        imageData = mat2gray(preprocessedImages{imageIdx});
        
        % Save the preprocessed image
        outputImagePath = fullfile(outputClassPath, sprintf('preprocessed_dwc_%s', imageFiles(imageIdx).name));
        imwrite(normalizedImage, outputImagePath);
    end
end