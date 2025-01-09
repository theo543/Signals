Use VLC Media Player or mpv player for MJPEG AVI file.
Windows 11 Media Player seems to cut width and height in half and mess up the colors (maybe it assumes chroma subsampling is used?).
See Media Player Glitch.mp4 for a screen recording of the issue.
Video source: https://www.pexels.com/video/green-bird-perched-on-tree-branch-4793475/

Program commands:
- examples (default)
  Encodes scipy.datasets.face in both color and greyscale using various quality settings.
  Also encodes using whatever quality settings produces a MSE of 15 (using binary search).
  If "Green Bird Perched on Tree Branch.avi" is missing, generates it as well, with a MSE of 15.
- image INPUT [-o OUTPUT] [--overwrite] [--quality Q] [--mse MSE]
  Encodes INPUT image to OUTPUT.
- video INPUT [-o OUTPUT] [--overwrite] [--quality Q] [--mse MSE]
  Encode INPUT video to OUTPUT.

Quality setting from 1 to 100 is implemented using the standard IJG quality algorithm: https://stackoverflow.com/a/29216609
MSE setting will use binary search to find the quality setting.
Default quality is 50.

Video quality setting is decided per-frame by randomly selecting a few blocks for binary search.
