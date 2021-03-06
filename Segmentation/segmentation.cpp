#include "opencv2\opencv.hpp"
#include "opencv2\ocl\ocl.hpp"

int main(int argc, char* argv[])
{
	//set up opencl contexts
	std::vector<cv::ocl::Info> ocl_devices;
	cv::ocl::getDevice(ocl_devices, cv::ocl::CVCL_DEVICE_TYPE_GPU);
	for (std::vector<cv::ocl::Info>::iterator it = ocl_devices.begin();
		it != ocl_devices.end();
		it++)
	{
		for (int i = 0; i < (*it).DeviceName.size(); i++)
		{
			std::cout << (*it).DeviceName[i] << std::endl;
		}

		break;
	}

	//setting the device 1
	cv::ocl::setDevice(ocl_devices[0], 0);

	if (argc < 2)
	{
		std::cout << "Specify an image file." << std::endl;
		return 1;
	}

	int spatial_radius = 10;
	if (argc > 2)
	{
		spatial_radius = atoi(argv[2]);
	}

	int color_radius = 20;
	if (argc > 3)
	{
		color_radius = atoi(argv[3]);
	}

	cv::Mat f = cv::imread(argv[1]);

	cv::cvtColor(f, f, CV_BGR2BGRA);

	cv::ocl::oclMat f_ocl(f);
	cv::Mat f_segmented;

	cv::ocl::meanShiftSegmentation(f_ocl, f_segmented, spatial_radius, color_radius, 20);

	cv::imshow("Image", f_segmented);
	cv::waitKey();

	return 0;
}