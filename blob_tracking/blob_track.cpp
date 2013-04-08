#include "opencv2\opencv.hpp"



int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Please specify a video file." << std::endl;
		return 1;
	}

	std::string video_file = argv[1];

	cv::VideoCapture cap(video_file.c_str());

	cv::Mat frame;

	cap >> frame;
	cv::Rect tracked_object(cv::Rect(650, 300, 150, 150));
	cv::Mat mask(frame.rows, frame.cols, CV_8UC1);
	cv::MatND hist;
	
	float** bounds = NULL;
	int hist_size[] = {50,50,50};
	int channels[] = {2};
	float cranges[] = {0, 256};
	const float* ranges[] = { cranges, cranges, cranges};
	
	cv::TermCriteria term_crit(CV_TERMCRIT_ITER, 100, .0001);

	while(1)
	{
		mask.setTo(0);
		mask(tracked_object) = 255;
		cv::calcHist(&frame, (int) 1, channels, mask, hist, 3, hist_size, ranges, true, false);

		//get new frame
		cap >> frame;

		if (frame.empty()) break;

		cv::Mat backproject(frame.rows, frame.cols, CV_32FC1);

		cv::calcBackProject(&frame, 1, channels, hist, backproject, ranges, 1, true);

		int niter = cv::meanShift(backproject, tracked_object, term_crit);

		std::cout << "Meanshift iterations: " << niter << std::endl;

		cv::rectangle(frame, tracked_object, cv::Scalar(255,0,0), 2, CV_AA);

		cv::imshow("test", frame);	
		
		cv::waitKey(30);

	}
	return 0;

}