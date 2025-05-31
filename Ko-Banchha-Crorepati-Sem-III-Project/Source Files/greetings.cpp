#include"greetings.hpp"
string greetings()
{
	std::string greeting;
	time_t currentTime; // get current time
	time(&currentTime);
	struct tm localTime;
	localtime_s(&localTime, &currentTime);
	int hour = localTime.tm_hour;
	if (hour >= 12 and hour < 17) // checks hour and prints greetings
		greeting = "Good Afternoon";
	else if (hour >= 17)
		greeting = "Good Evening";
	else
		greeting = "Good Morning";
	return greeting;
}