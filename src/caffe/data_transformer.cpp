#include <string>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
const int DataTransformer<Dtype>::widths_[] = {256, 256, 192, 224, 224, 168, 168, 168, 126};

template<typename Dtype>
const int DataTransformer<Dtype>::heights_[]  = {256, 192, 256, 224, 168, 224, 168, 126, 168};

template<typename Dtype>
void DataTransformer<Dtype>::TransformSingle(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
                                       vector<int>& bbox) {
  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  int channels = img->nChannels;
  int width = img->width;
  int height = img->height;
  unsigned char* data = (unsigned char *)img->imageData;
  int step = img->widthStep / sizeof(char);
  // crop 4 courners + center
  int w[5], h[5];
  FillInOffsets(w, h, width, height, crop_size);
  int h_off, w_off;
  // We only do random crop when we do training.
  if (Caffe::phase() == Caffe::TRAIN) {
    int r = Rand() % 5;
    h_off = h[r];
    w_off = w[r];
  } else {
    h_off = h[4];
    w_off = w[4];
  }

  ////// -------------------!! for debug !! -------------------
  // IplImage *dest = cvCreateImage(cvSize(crop_size * 2, crop_size * 2),
                                    // img->depth, img->nChannels);
  // cvResize(img, dest);
  // cvNamedWindow("Sample1");
  // cvNamedWindow("Sample2");
  // if (phase_ == Caffe::TRAIN)
  //   cvShowImage("Sample", dest);
  // else
  //   cvShowImage("Sample", img);
  // cvWaitKey(0);
  // cvReleaseImage(&img);
  // cvReleaseImage(&dest);
  // if (phase_ == Caffe::TRAIN) {
  //   cvSetImageROI(img, cvRect(w_off, h_off, crop_size, crop_size));
  //   // cvCopy(img, dest, NULL);
  //   cvResize(img, dest);
  //   cvResetImageROI(img);
  //   cvShowImage("Sample1", img);
  //   cvShowImage("Sample2", dest);
  //   cvWaitKey(0);
  // }
  // cvReleaseImage(&dest);
  ////// -------------------------------------------------------
  bool is_mirror = mirror && Rand() % 2;
  if (is_mirror) {
    // Copy mirrored version
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + (crop_size - 1 - w);
          int data_index = (h + h_off) * step + (w + w_off) * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }

  } else {
    // Normal copy
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + w;
          int data_index = (h + h_off) * step + (w + w_off) * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }
  }

  // change the bbox according to croping
  if (bbox.size() > 0){
	  // has bbox info
	  for (int i = 0; i < bbox.size()/2; i++){

		  bbox[2 * i] = min(crop_size, max(0, bbox[2 * i] - w_off));
		  bbox[2 * i + 1] = min(crop_size, max(0, bbox[2 * i + 1] - h_off));
	  }
		if (is_mirror) {
			// mirror the bbox horizontally
			bbox[0] = crop_size - bbox[0] - 1;
			bbox[2] = crop_size - bbox[2] - 1;
		}
  }

}

template<typename Dtype>
void DataTransformer<Dtype>::TransformMultiple(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
                                       vector<int>& bbox) {
  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  int channels = img->nChannels;
  int width = img->width;
  int height = img->height;


  int sc = 3;  //(224, 224)
  int cr = 4;  // center crop
  // We only do random cropping & scaling when we do training.
  if (Caffe::phase() == Caffe::TRAIN) {
    sc = Rand() % 9;
    cr = Rand() % 5;
  }
  int roi_w = widths_[sc];
  int roi_h = heights_[sc];
  // crop 4 courners + center
  int w[5], h[5];
  FillInOffsets(w, h, width, height, roi_w, roi_h);
  int h_off = h[cr], w_off = w[cr];
  IplImage *dest = cvCreateImage(cvSize(crop_size, crop_size),
                                    img->depth, img->nChannels);

  cvSetImageROI(img, cvRect(w_off, h_off, roi_w, roi_h));
  cvResize(img, dest); // crop and resize to crop_size * crop_size
  cvResetImageROI(img);

  //////--------------------!! for debug only !!-------------------
  // if (phase_ == Caffe::TRAIN) {
  //   cvFlip(dest, NULL, 1);
  //   cvShowImage("Sample1", img);
  //   cvShowImage("Sample2", dest);
  //   LOG(INFO) << w_off << ", " << h_off << "   " << roi_w << ", " << roi_h;
  //   cvWaitKey(0);
  // }

  unsigned char* data = (unsigned char *)dest->imageData;
  int step = dest->widthStep / sizeof(char);

  bool is_mirror = mirror && Rand() % 2;
  if (is_mirror) {
    // Copy mirrored version
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + (crop_size - 1 - w);
          int data_index = h * step + w * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }
  } else {
    // Normal copy
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + w;
          int data_index = h * step + w * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }
  }
  cvReleaseImage(&dest);

  // change the bbox according to croping
    if (bbox.size() > 0){
  	  // has bbox info
    	for (int i = 0; i < bbox.size()/2; i++){

    		bbox[2 * i] = min(roi_w, max(0, bbox[2 * i] - w_off));
    		bbox[2 * i + 1] = min(roi_h, max(0, bbox[2 * i + 1] - h_off));
    	}
		if (is_mirror) {
			// mirror the bbox horizontally
			bbox[0] = roi_w - bbox[0] - 1;
			bbox[2] = roi_w - bbox[2] - 1;
		}
		// resize the bounding box according to the aspect ratio and crop_size
		for (int i = 0; i < bbox.size() / 2; i++) {

			bbox[2 * i] = (bbox[2 * i] * crop_size) / roi_w;
			bbox[2 * i + 1] = (bbox[2 * i + 1] * crop_size) / roi_h;
		}
	}

}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
                                       vector<int>& bbox) {
  if (!param_.multiscale())
    TransformSingle(batch_item_id, img, mean, transformed_data,
    		bbox);
  else
    TransformMultiple(batch_item_id, img, mean, transformed_data,
    		bbox);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (Caffe::phase() == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (Caffe::phase() == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
