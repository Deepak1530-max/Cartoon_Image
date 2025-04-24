import cv2

class Cartoonizer:
    """
    Cartoonizer effect
    A class that applies a cartoon effect to an image.
    The class uses a bilateral filter and adaptive thresholding to create
    a cartoon effect.
    """
    def __init__(self):
        pass

    def render(self, img_path):
        img_rgb = cv2.imread(img_path)
        img_rgb = cv2.resize(img_rgb, (1366, 768))

        numDownSamples = 2  # Number of downscaling steps
        numBilateralFilters = 50  # Number of bilateral filtering steps

        # -- STEP 1 -- Downsample and apply bilateral filter
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # -- STEPS 2 and 3 -- Convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        # -- STEP 4 -- Detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 -- Combine edge mask with the color image
        img_edge = cv2.resize(img_edge, (img_color.shape[1], img_color.shape[0]))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        cartoon = cv2.bitwise_and(img_color, img_edge)
        return cartoon

# Usage
if __name__ == "__main__":
    file_name = r"Screenshot.jpg"  # Replace with your image file
    cartoonizer = Cartoonizer()
    cartoon_img = cartoonizer.render(file_name)

    # Save and show output
    cv2.imwrite("Cartoon_Version.jpg", cartoon_img)
    cv2.imshow("Cartoonized Image", cartoon_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
