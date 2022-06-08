#include <ctime>
#include <thread>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>

#define UTC (+3)

using namespace std;
using namespace cv;

class Screenshotter
{
public:
static int getTime();
static int getDate();
static void takeScreenshot(useconds_t sleepTime);
};


int Screenshotter::getTime()
{
time_t currentTime;

struct tm * pointerTM;

time ( &currentTime );

pointerTM = gmtime( &currentTime );

return printf ("Time in sec: %02d\n", (pointerTM->tm_sec));
}

int Screenshotter::getDate()
{
    time_t timenow = time(0);
    struct tm * datenow = localtime( &timenow );

    return printf ("date, month, year: %02d-%02d-%02d\n", (datenow->tm_mday), (datenow -> tm_mon), (datenow -> tm_year +1900));
}
    
void Screenshotter::saveScreenshot(useconds_t sleepTime)
{
    string DateMonthYear = to_string(Screenshotter::getDate());
    string pathToSaveImage = "/Users/neilsuji/Downloads/screenshot_images/";
    Mat imageSave = imwrite(pathToSaveImage+".png", "image from screenshotter goes here" );
    bool check = imageSave;
}

void Screenshotter::takeScreenshot()
{

}
int main(){
    //TO DO: invoke getTime function and screenshot function exactly once per minute
    while (true){
        usleep(6000000000 - Screenshotter::getTime() % 6000000000)
        Screenshotter::takeScreenshot(60000000);
    };
    return 0;
}
