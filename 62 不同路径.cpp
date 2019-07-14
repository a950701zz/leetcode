#include<vector>
#include <fstream>
#include<iostream>
#include<map>
#include<string.h>
using namespace std;
class Solution {
public:
    int nums[101][101];
    int uniquePaths(int m, int n) {
        memset(nums,0,sizeof(nums));
        return uniquePaths1(m,n);
    }
    
     int uniquePaths1(int m, int n) {
    	
        if(m==1|| n==1) {nums[m][n]=1;return 1;}
        
        if(m==2&&n==2) {
        	nums[m][n]=2;return 2;
        }
         if(nums[m][n]!=0) {
             return nums[m][n];
         }
         nums[m-1][n] =    uniquePaths1(m-1,n);
         nums[m][n-1] =    uniquePaths1(m,n-1);
         return nums[m-1][n]+nums[m][n-1];
    }
};

int main(int argc, char* argv[])
{
    Solution t;
    cout<<t.uniquePaths(7,3);
}
