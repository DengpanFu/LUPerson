# The Process to build LUPerson

All the videos' YouTube key can be found at [vnames.txt](https://drive.google.com/file/d/1eopcxZPNHnaobjnSwP37U2YFlptNE0A0/view?usp=sharing)

All detection results can be found at [dets.pkl](https://drive.google.com/file/d/1t_XPHOI_VzuebaAnXccu4iIzDXMNpTJO/view?usp=sharing)

**!! The following scripts are not well tested, but provide the main processes !!**.

## Download the raw videos
```
python download.py -f ${YOUR_VIDEO_NAME_FILE_DIR}/vname.txt -s ${YOUR_VIDEO_DIR}
```
[youtube-dl](https://github.com/ytdl-org/youtube-dl) is needed.

## Extract images from raw videos and their detections
```
python extract.py -v ${YOUR_VIDEO_DIR} -d ${DETECTION_DIR} -s ${SAVE_DIR}
```

## Convert extracted images to lmdb data
```
python convert_lmdb.py
```

# Three is a third reconstruction at [Issue](https://github.com/DengpanFu/LUPerson/issues/8#issuecomment-1004611808), please refer to it.
