def mean_square_error(orig_img, output_img):
    diff = orig_img - output_img
    sq_diff = diff ** 2
    return sq_diff.mean()
