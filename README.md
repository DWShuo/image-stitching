## Usage

``python stitch.py <img_dir>``

Script takes a folder of images as its input

## Example image

original images side by side

![original images](./image2-image3.png)

Here are the keypoints detected after the inital FLANN matching

![FLANN matching](./image2_image3_match_init.jpg)

Inlier keypoints after Fundamental matrix estimation

![Fundamental matrix estimation](./image2_image3_match_fund.jpg)

Remaining keypoints after homography estimation

![Homography matrix](./image2_image3_match_homo.jpg)

Finally here are the two images stitched together along with the output log

![Final image](./image2_image3.jpg)

## Output log

```
Comparing image2.JPG and image3.JPG
Matches found: 1287
Inliers count after Fundamental estimate: 966
Fundamental decision ---
Matched scene: inlier threshold meet
Inliers count after Homography estimate 397
Homography decision ---
image2_image3 Possible for alignment
Alignment possible: combine images
Warp Image 1 -> Image 2
```
