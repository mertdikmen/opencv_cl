#include "opencv2/opencv.hpp"
#include "opencv2\ocl\ocl.hpp"
//#include "opencv2\objdetect\objdetect.hpp"

int main(int argc, char* argv[])
{
	//set up opencl contexts
	std::vector<cv::ocl::Info> ocl_devices;
	cv::ocl::getDevice(ocl_devices);

	for (std::vector<cv::ocl::Info>::iterator it = ocl_devices.begin();
		it != ocl_devices.end();
		it++)
	{
		std::cout << it->DeviceName[0] << std::endl;
	}

	if (argc < 2)
	{
		std::cout << "Specify a video file." << std::endl;
		return 1;
	}


	std::cout << "Reading video file: " << argv[1] << std::endl;

	cv::VideoCapture cap(argv[1]);

	double n_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);

	std::cout << "Number of frames: " << n_frames << std::endl;

	int start_frame = 0;
	if (argc > 2)
	{
		start_frame = atoi(argv[2]);
		cap.set(CV_CAP_PROP_POS_FRAMES, start_frame);
	}

	int n_process = 100;
	if (argc > 3)
	{
		n_process = atoi(argv[3]);
	}

	cv::ocl::OclCascadeClassifier cascade;

	cascade.load("haarcascade_frontalface_alt.xml");

	cv::Mat f;
	cv::Mat f_gray;

	std::vector<cv::Rect> detections;

	for (int i = 0; i < n_process; i++)
	{
		cap >> f;

		cv::cvtColor(f, f_gray, CV_BGR2GRAY);

		cascade.detectMultiScale(f_gray, detections, 1.1, 3, 0, cv::Size(75, 75));

		if (detections.size() > 0)
		{
			int largest_area = 0;
			cv::Rect largest_rect(0, 0, 0, 0);

			for (std::vector<cv::Rect>::iterator it = detections.begin();
				it != detections.end();
				it++)
			{
				if (it->area() > largest_area)
				{
					largest_area = it->area();
					largest_rect = *it;
				}
			}

			cv::rectangle(f, largest_rect, cv::Scalar(255, 0, 0), 2, CV_AA);
		}
		cv::imshow("video", f);
		cv::waitKey(10);
	}

	return 0;
}