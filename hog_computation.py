from skimage.feature import hog

def get_hog_features(gray, orientations=8, pixels_per_cell=4, cells_per_block=2, transform_sqrt=True, visualise=True):
    if visualise:
        return hog(gray,
                   orientations=orientations,
                   pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block),
                   transform_sqrt=transform_sqrt,
                   visualise=True)
    else:
        return hog(gray,
                   orientations=orientations,
                   pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block),
                   transform_sqrt=transform_sqrt,
                   visualise=False)