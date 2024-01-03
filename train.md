```sh
$ # Copy the images to tesstrain/data/MODEL_MAME-ground-truth with the following structure:
$ # tesstrain/data/MODEL_MAME-ground-truth/MODEL_MAME-ground-truth.gt.txt containing the text
$ # tesstrain/data/MODEL_MAME-ground-truth/MODEL_MAME-ground-truth.png containing the image

$ cd tesstrain
$ gmake training MODEL_NAME=MODEL_MAME
$ copy MODEL_MAME.traineddata to tessdata
$ # paste tesstrain/data/MODEL_MAME.traineddata to tessdata (/opt/homebrew/share/tessdata/ on mac)
$ cp MODEL_MAME.traineddata /opt/homebrew/share/tessdata/
```