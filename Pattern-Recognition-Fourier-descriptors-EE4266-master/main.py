import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.medianBlur(imgray, 5)
    ret, thresh = cv2.threshold(
        imgray, 140, 255, cv2.THRESH_BINARY_INV)  # Binarizing the image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Detecting contours
    return contours


def two_d_to_complex(contour):
    x = [x[0][0] for x in contour]
    y = [y[0][1] for y in contour]
    complex_form = np.array(x)+np.array(y)*1j
    return complex_form


def eliminate_higher_freq(FT_array, keep_fraction):
    z = np.copy(FT_array)
    n = len(z)
    # a[0] represents the DC component. a [1: n//2] represent positive frequencies
    # from lowest to highest. a[n//2+1:] represents negative frequencies from most
    # negative to least negative
    z[int(keep_fraction*n/2)+1:-int(keep_fraction*n/2)] = 0
    return z


def plot_points_on_image(img_name, smooth_complex_array):
    im = plt.imread(img_name)
    implot = plt.imshow(im)
    plt.scatter(x=smooth_complex_array.real, y=smooth_complex_array.imag, s=3)
    plt.show()


def show_all_contours(img, contours):
    out = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    out = cv2.resize(out, (543, 700))
    cv2.imshow('output', out)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


def get_template_value(template, keep_fraction):
    contours = detect_contours(template)
    complex_array = two_d_to_complex(contours[0])
    FT_array = np.fft.fft(complex_array)
    FT_array = eliminate_higher_freq(FT_array, keep_fraction)
    return FT_array


def normalise_array(FT_array):
    # Absolute value: makes analysis rotaion invariant
    FT_array = np.absolute(FT_array)
    # Remove zero frequency: Make translation invariant
    FT_array = FT_array[1:]
    # Divide each element by first element: Make scale invariant
    FT_array /= FT_array[0]
    return FT_array


def resize(array, num_elements):
    array = np.append(array[1:num_elements], array[-num_elements])
    # array = array[:num_elements]
    return array


def ssd(normalised_template_array, normalised_array):
    return np.sum(np.square(normalised_array - normalised_template_array))


img_name = 'a4.bmp'
template_name = 'C.bmp'
img = cv2.imread(img_name)
template = cv2.imread(template_name)
keep_fraction = 0.15

template_array = get_template_value(template, keep_fraction)
normalised_template_array = normalise_array(template_array)
normalised_template_array = resize(normalised_template_array, 10)
print(normalised_template_array)

contours = detect_contours(img)

# Remove small contours
lengths = [len(contour) for contour in contours]
lengths.sort()
contours = sorted(contours, key=lambda x: len(x))
contours = contours[-50:]


desired_contours = []
for contour in contours:
    complex_array = two_d_to_complex(contour)
    FT_array = np.fft.fft(complex_array)
    FT_array = eliminate_higher_freq(FT_array, keep_fraction)
    normalised_array = normalise_array(FT_array)
    normalised_array = resize(normalised_array, 10)
    diff = ssd(normalised_template_array, normalised_array)
    if(diff < 0.03):
        desired_contours.append(contour)


# differences = enumerate(differences)
# differences = sorted(differences, key=lambda x: x[1])
# print(differences)

# contour_indices = [x[0] for x in differences[:5]]

# desired_contours = [contours[i] for i in set(contour_indices)]
# plot_points_on_image(template_name, template_array)

out = cv2.drawContours(img, desired_contours, -1, (0, 255, 0), 3)
out = cv2.resize(out, (543, 700))
cv2.imshow('output', out)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

# show_all_contours(img, contours)
