#include <vector>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <algorithm>    // std::max
#include "string"
#include <set>
#include<queue>
#include <stack>
using namespace std;
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
//                                     1
//
//class Solution
//{
//public:
//	vector<int> twoSum(vector<int>& nums, int target)
//	{
//		vector<int> ans;
//		map<int, int> hashT;
//		int size = nums.size();
//		map<int, int> ::iterator it;
//		for (int i = 0; i < size; i++)
//		{
//			if ((it = hashT.find(target - nums[i])) != hashT.end())
//			{
//				ans.push_back(it->second);
//				ans.push_back(i);
//				return ans;
//			}
//			else
//				hashT.insert(map<int, int>::value_type(nums[i], i));
//		}
//	}
//};

//                                     3
//
//class Solution
//{
//public:
//	vector<int> twoSum(vector<int>& nums, int target)
//	{
//		vector<int> ans;
//		map<int, int> hashT;
//		int size = nums.size();
//		map<int, int> ::iterator it;
//		for (int i = 0; i < size; i++)
//		{
//			if ((it = hashT.find(target - nums[i])) != hashT.end())
//			{
//				ans.push_back(it->second);
//				ans.push_back(i);
//				return ans;
//			}
//			else
//				hashT.insert(map<int, int>::value_type(nums[i], i));
//		}
//	}
//};
//
//class Solution {
//public:
//	int lengthOfLongestSubstring(string s) {
//		int rst = 1;
//		if (s.length() == 0) return 0;
//		for (int i = 0; i < s.length(); i++)
//		{
//			for (int j = i + rst; j <= s.length()-1; j++)
//			{
//				string sub = s.substr(i, j-i+1);  //字符串长度j+1-i
//				if (allUnique(sub) == true)
//				{
//					int a = sub.length();
//					rst = max(rst, j - i + 1);
//				}
//				else break;
//			}
//		}
//		return rst;
//	}
//	
//	
//	bool allUnique(string s)
//	{
//		for (int i = 0; i < s.length()-1; i++)
//		{
//			for (int j = i+1; j < s.length(); j++)
//			{
//				if (s[i] == s[j]) return false;
//			}
//		}
//		return true;
//	}
//};


//7
//class Solution {
//public:
//	int reverse(int x) {
//		int a, b; 
//		int  rst = 0;
//		while (x / 10 != 0)
//		{
//			int pop = x % 10;
//			if (rst > INT_MAX / 10 || (rst == INT_MAX / 10 && pop > 7)) return 0;
//			if (rst < INT_MIN / 10 || (rst == INT_MIN / 10 && pop < -8)) return 0;
//			rst = rst * 10 + pop;
//			x = x / 10;
//		}
//		if (rst > INT_MAX / 10 || (rst == INT_MAX / 10 && x % 10 > 7)) return 0;
//		if (rst < INT_MIN / 10 || (rst == INT_MIN / 10 && x % 10 < -8)) return 0;
//		rst = rst * 10 + x % 10;
//		return rst ;
//		
//	}
//};


//35 普通遍历法
//class Solution {
//public:
//	int searchInsert(vector<int>& nums, int target) {
//		if (nums.size() == 1)
//		{
//			if (target <= nums[0]) return 0;
//			else
//				return 1;
//		}
//		if (nums.size() == 2)
//		{
//			if (target <= nums[0]) return 0;
//			else
//				if (target > nums[0] && target <= nums[1]) return 1;
//				else
//					return 2;
//		}
//		for (int i = 1; i <= nums.size()-2;i++)
//		{
//			if (target <= nums[i - 1])
//				return i - 1;
//			if (target > nums[i - 1] && target <= nums[i])
//				return i;
//		}
//		if (target <= nums[nums.size() - 1]) return nums.size() - 1;
//		else
//			return nums.size();
//	}
////};
//
//
//	//二分法查找
//
//class Solution {
//	public:
//		int searchInsert(vector<int>& nums, int target) {
//
//			/*if (nums.size() == 0 || nums[0] > target)
//				return 0;
//			if (nums[nums.size() - 1] < target)
//				return nums.size();*/
//			int low = 0; int hight = nums.size() - 1;
//			int mid = 0;
//			while (low<=hight)
//			{
//				mid = (low + hight) / 2;
//				if (nums[mid]<target)
//				{
//					low = mid + 1;
//				}
//				else if (nums[mid]>target)
//				{
//					hight = mid - 1;
//				}
//				else
//					return mid;
//			}
//
//			return nums[mid] >= target ? mid : mid + 1;
//		}
//	};


//70爬楼梯
//class Solution {
//public:
//	int climbStairs(int n) {
//		if (n < 1) return 1;
//		vector<int> dp(n+100);
//		dp[0] = 1; dp[1] = 1;
//		for (int i = 2; i <=n; i++)
//		{
//			dp[i] = dp[i - 1] + dp[i - 2];
//		}
//		return dp[3];
//	}
//};

// 53
//class Solution {
//public:
//	int maxSubArray(vector<int>& nums) {  //maxsum 表示包含下标为i的元素的子串的最大的数
//		int ans = nums[0];
//		int maxsum = nums[0];
//		for (int i = 1; i < nums.size();i++)
//		{
//			maxsum = maxsum + nums[i] > nums[i] ? maxsum + nums[i] : nums[i];
//			ans = maxsum > ans ? maxsum : ans;
//		}
//		return ans;
//	}
//};

//     121 
//class Solution {
//public:
//	int maxProfit(vector<int>& prices) {
//		int minprice = min(prices[0],prices[1]);
//		int maxprofit = prices[1] - prices[0] > 0 ? prices[1] - prices[0] : 0;
//		for (int i = 2; i < prices.size();i++)
//		{
//			if (minprice > prices[i-1]) minprice = prices[i-1]; //前一天的最小价格
//			maxprofit = max(maxprofit, prices[i] - minprice );
//
//		}
//		return maxprofit<0?0:maxprofit;
//	}
//};


// 198
//class Solution {
//public:
//	int rob(vector<int>& nums) {
//		if(nums.size() == 1) return nums[0];
//		if (nums.size() == 2) return max(nums[0],nums[1]);
//		if (nums.size() == 3) return max(nums[0]+nums[2], nums[1]);
//		if (nums.size() == 4) return max(nums[0] + nums[2], nums[1]);
//		
//		int current3 = nums[2] + nums[0];
//		int current2 = nums[1];
//		int current1 = nums[0];
//		int current4 = max(current2 + nums[3], current1 + nums[3]);
//		int middle = current4;
//
//		for (int i = 4; i < nums.size();i++)
//		{
//			current1 = current2;
//			current2 = current3;
//			current3 = current4;
//			current4 = max(current2 + nums[i], current1 + nums[i]);
//		
//		}
//		return max(current4,current3);
//
//	}
//};


//764
//class Solution {
//public:
//	int minCostClimbingStairs(vector<int>& cost) {
//		// rst = min{dp[i],dp[i-1]}
//		int currenti = 0;
//		int currenti_1 = 0;
//		int rst = 0;
//		for (int i = 2; i <= cost.size() - 1; i++)
//		{
//			rst = min(currenti + cost[i - 1], currenti_1 + cost[i - 2]);
//			currenti_1 = currenti;
//			currenti = rst;
//		}
//		int index = cost.size()-1;
//		return min(currenti + cost[index], currenti_1 + cost[index - 1]);
//	}
//};

//   169
//class Solution {
//public:
//	int majorityElement(vector<int>& nums) {
//		int current = nums[0];
//		int count = 0;
//		for (int i = 0; i <= nums.size()- 1;i++)
//		{
//			if (count == 0)
//			{
//				current = nums[i];
//				count++;
//			}
//			else if (current == nums[i])
//			{
//				count++;
//			}
//			else if (current != nums[i])
//			{
//				count = count - 1;
//			}
//		}
//		return current;
//	}
//};

// 完全平方数
//class Solution {
//public:
//	int numSquares(int n) {
//		if (n == 1) return 1;
//		if (n == 2) return 2;
//		if (n == 0) return 0;
//		vector<int> f(n+1,INT_MAX);  //
//		f[0] = 0; f[1] = 1; f[2] = 2;
//		for (int i = 3; i <= n; i++)
//		{
//			for (int j = sqrt(i); j>=1; j--)
//			{
//				if (i == j*j)
//				{
//					f[i] = 1; break;
//				}
//				else
//				f[i] = min(f[i], f[j*j] + f[i-j*j]);  //如果没算过f[i+j*j]则等于f[i]+1
//			}
//		}
//		return f[n];
//	}
//};
//
//class Solution {
//public:
//	int maxProduct(vector<int>& nums) {
//		if (nums.size() == 1) return nums[0];
//		if (nums.size() == 2) return max(max( nums[0], nums[1]), nums[0] * nums[1] );
//		int current = max( nums[0] * nums[1], nums[1] );
//		int current_ = min(nums[0] * nums[1], nums[1]);
//		int previous = nums[0];
//		int previous_ = nums[0];
//		int rst = max(current, previous);
//		for (int i = 2; i <= nums.size() - 1; i++)
//		{
//			previous = current;
//			previous_ = current_;
//			current = max(max(previous*nums[i], nums[i]),previous_*nums[i]);
//			current_ = min(min(previous*nums[i], nums[i]), previous_*nums[i]);
//			if (rst < current) rst = current;
//		
//		}
//		return rst;
//	}
//};


////旋转数组
//class Solution {
//public:
//	void rotate(vector<int>& nums, int k) {
//		int sum = nums.size();
//		vector <int> copyed(nums);
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			nums[(i + k)%sum] = copyed[i];
//		}
//	}
//};


//移动零
//class Solution {
//public:
//	void moveZeroes(vector<int>& nums) {
//	 
//		int count = 0; int sizes = nums.size();
//		for (int i = 0; i <= sizes-1;i++)
//		{
//			if (nums[i]==0)
//			{
//				count++;
//			}
//			else
//			{
//				nums[i - count] = nums[i];
//			}
//		}
//		for (int i = 0; i < count;i++)
//		{
//			nums[sizes - i - 1] = 0;
//		}
//	}
//};

//求交集
//class Solution {
//public:
//	vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
//		map<int, int> map1; int size1 = nums1.size();
//	    int size2 = nums2.size();
//		vector<int> result;
//		for (int i = 0; i <= size1 - 1; i++)
//		{
//			map1[nums1[i]]++;
//		}
//		
//
//		for (int i = 0; i <= size2 - 1; i++)
//		{
//			if (map1.count(nums2[i]) == 0||map1[nums2[i]]==0)
//				continue;
//			else
//			{
//				result.push_back(nums2[i]);
//				map1[nums2[i]]=0;
//			}
//		}
//		return result;
//	}
//
//	
//};
//
//class Solution {
//public:
//	bool searchMatrix(vector<vector<int>>& matrix, int target) {
//		//if (matrix == [[]]) return true;
//		if (matrix.size() == 0) return false;
//		if (target <= matrix[0][0]) return false;
//		int rst = 0;
//		for (int i = 1; i <= matrix.size() - 1; i++)
//		{
//			if (target < matrix[i][i])
//			{
//				rst = i;
//				break;
//			}
//			else
//				if (target > matrix[i][i]) {
//					rst = i;
//					continue;
//				}
//				else
//					if (target == matrix[i][i]) return true;
//		}
//		if (rst == matrix.size() - 1) return false;
//	
//		//若没有返回，接下来只用从第i行和第i列中筛选
//		//第i行
//		
//		for (int i = 0; i <= rst; i++)
//		{
//			if (target < matrix[rst][i]) break;
//			if (target == matrix[rst][i]) return true;
//			if (target > matrix[rst][i]) continue;
//		}
//
//		for (int i = 0; i <= rst; i++)
//		{
//			if (target < matrix[i][rst]) break;
//			if (target == matrix[i][rst]) return true;
//			if (target > matrix[i][rst]) continue;
//		}
//		return false;
//
//	}
//};

//  最大数
//class Solution {
//public:
//	string largestNumber(vector<int>& nums) {
//		BubbleSort(nums); string rst;
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			rst += to_string(nums[i]);
//		}
//		if (rst[0] == '0') return "0";
//		return rst;
//	}
//	void BubbleSort(vector<int> &nums)
//	{
//		for(int i = 0; i < nums.size()-1; i++)
//		{
//			for (int j = 0; j <= nums.size()-2- i;j++)
//			{
//				if (isminus(nums[j], nums[j + 1]))
//				{
//					int temp = nums[j+1];
//					nums[j + 1] = nums[j];
//					nums[j] = temp;
//				}
//			}
//		}
//	}
//	bool isminus( int a, int b )
//	{
//		return to_string(a) + to_string(b) < to_string(b) + to_string(a);
//	}
//};

//只出现一次的数
//class Solution {
//public:
//	int singleNumber(vector<int>& nums) {
//		int rst = 0;
//		for (int i = 0; i < nums.size(); i++)
//			rst ^= nums[i];
//		return rst;
//	}
//};

//买股票的最佳时机2
//class Solution {
//public:
//	int maxProfit(vector<int>& prices) {
//		int profit = 0;
//		for (int i = 0; i < prices.size() - 1; i++)
//		{
//			profit += prices[i + 1] - prices[i]>0 ? prices[i + 1] - prices[i] : 0;
//		}
//		return profit;
//	}
//};

//分发饼干
//class Solution {
//public:
//	int findContentChildren(vector<int>& g, vector<int>& s) {
//		sort(g.begin(), g.end()); int count = 0;
//		sort(s.begin(), s.end()); int i = 0; int j = 0;
//		if (g.size() == 0 || s.size() == 0) return 0;
//
//		for (; i <= g.size() - 1 && j <= s.size() - 1;)
//		{
//			if (s[j] >= g[i])
//			{
//				count++; i++; j++;
//			}//当前饼干可以分发
//			else
//				j++;
//		}
//		return count;
//	}
//};

//分数到小数
//class Solution {
//public:
//	string fractionToDecimal(int numerator, int denominator) {
//		long long t = numerator, d = denominator;
//		map<long  long,int > A;  //建立一个Map A记录的是当前的除数以及除数出现是当时ans的size,为了在下一次出现
//		                        //同样的除数时会开始循环，记录他的size确定（插入的位置。
//		string ans;
//		if (t*d < 0) ans = "-";
//		t = abs(t); d = abs(d);
//		ans += to_string(t / d);  //整数部分
//		t = t%d;  //小数部分取余数
//		if (!t) return ans;
//		ans += ".";
//		while (t)   //当除数为0的时候循环结束 t是当前除以分母的数
//		{
//			if (A.count(t))  //当重复的除数出现是停止循环
//			{
//				ans.insert(A[t], "(");
//				ans.push_back(')');
//				return ans;
//			}
//			A[t] = ans.size();
//			ans += '0' + t * 10 / d;
//			t = t * 10 % d;
//		}
//		return ans;
//	}
//};

//缺失的数字
//class Solution {
//public:
//	int missingNumber(vector<int>& nums) {
//		int sum = nums.size();
//		for (int i = 0; i <= nums.size() - 1; i++)
//		{
//			sum ^= nums[i];
//			sum ^= i;
//		}
//		return sum;
//		
//	}
//};

//位1的个数   一个数和他减去1相与得到的结果是把他最右边的1变成0
//class Solution {
//public:
//	int hammingWeight(uint32_t n) {
//		int count = 0;
//		while (n)
//		{
//			n &= (n - 1);
//			count++;
//		}
//		return count;
//	}
//};


//阶乘后的0
//class Solution {
//public:
//	int trailingZeroes(int n) {
//		long long x = n; int i = 1;
//		long long y = 5, num = 0;
//		while (y <= x)
//		{
//			num++;
//			i++;
//			y = i * 5;
//		}
//		return num;
//	}
//};

//盛水最多的容器
//class Solution {
//public:
//	int maxArea(vector<int>& height) {
//		int i = 0, j = height.size() - 1; int result = 0;
//		while (i<j)
//		{
//			if (height[i] < height[j])
//			{
//				result = max(result, (j - i)*height[i]);
//				i++;
//			}
//			else
//			{
//				result = max(result, (j - i)*height[j]);
//				j--;
//			}
//		}
//		return result;
//	}
//};


//最长共同前缀
//class Solution {
//public:
//	string longestCommonPrefix(vector<string>& strs) {
//		if (strs.size() == 1) return strs[0];
//		if (strs.size() == 0) return "";
//		int mark = 0; int pol = 1;
//		for (int i = 0; i < strs.size() - 1; i++)
//			if (strs[i].size() == 0) return "";
//		for (int i = 0; i <= strs[0].size() - 1; i++)
//		{
//			for (int j = 1; j <= strs.size() - 1;j++)
//			{
//				
//				if (strs[0][i]!=strs[j][i])   
//				{
//					pol = -1;
//					mark = i;
//					goto here;
//				}
//			}
//			mark = i;
//		}
//
//		here:
//		if (mark == 0&&pol<0 )
//			return "";
//		else if (pol < 0) return strs[0].substr(0, mark);
//			else 
//			return strs[0].substr(0, mark+1);
//	}
//};
//
//class Solution {
//public:
//	vector<int> productExceptSelf(vector<int>& nums) {
//		vector <int> res(nums.size(), 1);
//		int num = 1;
//		for (int i = nums.size() - 1; i >= 1;i--)  //res数组中先存储当前位置右边数的乘积
//		{
//			res[i - 1] = num * nums[i];
//			num *= nums[i];
//		}
//		num = 1;
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			res[i] *= num;
//			num *= nums[i];
//		}
//		return res;
//	}
//};

//接雨水
//class Solution {
//public:
//	int trap(vector<int>& height) {
//		if (height.size() == 1 || height.size() == 0) return 0;
//		int MaxNum = 0; int index = 0;
//		for (int i = 0; i < height.size(); i++)
//		{
//			if (height[i] > MaxNum)
//			{
//				MaxNum = height[i];
//				index = i;
//			}
//		}
//		int last = height[0]; int drop = 0;
//		for (int i = 1; i < index; i++)
//		{
//			
//			if (height[i] < last)  //下一个柱子高度降低
//			{
//				drop += last - height[i];
//			}
//			if (height[i] >= last)
//			{
//				last = height[i]; continue;
//			}
//		}
//		last = height[height.size() - 1];
//		for (int i = height.size() - 2; i > index;i--)
//		{
//			if (height[i] < last)
//			{
//				drop += last - height[i];
//			}
//			if (height[i] >= last)
//			{
//				last = height[i]; continue;
//			}
//		}
//		return drop;
//	}
//};


//硬币找零
//class Solution {
//public:
//	int coinChange(vector<int>& coins, int amount) {
//		vector<int>dp(amount + 1, amount + 1);
//		dp[0] = 0;
//		if (coins.size() == 0) return 0;
//		for (int i = 1; i <= amount;i++)
//		{
//			for (int j = 0; j <= coins.size() - 1; j++)
//			{
//				if (i >= coins[j])    //i是当前动态规划的状态，遍历coins[j]每个硬币
//					dp[i] = min(dp[i], dp[i - coins[j]] + 1);
//				else
//					continue;
//			}
//		}
//		return dp[amount] == amount + 1 ? -1:dp[amount];
//	}
//};
//
//class Solution {
//public:
//	int longestConsecutive(vector<int>& nums) {
//		map<int, int> x; int rst = 1;
//		if (nums.size() == 0)return 0;
//		for (int i = 0; i < nums.size();i++)
//		{
//			int num = x.count(nums[i]);
//			if (num == 0)  //第一次遍历时
//			{
//				if (x.count(nums[i]-1)!=0)//左边数存在
//				{
//					if (x.count(nums[i]+1)!=0)  //左右边数都存在
//					{
//						int newsequence = x[nums[i] - 1] + x[nums[i] + 1] + 1;
//						cout << (nums[i] - 1);
//						cout << x.count(nums[i] - 1);
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i] + x[nums[i] + 1]] = newsequence;
//						x[nums[i] ] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //左边数存在，右边不存在
//					{
//						int newsequence = x[nums[i] - 1] + 1;
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//				}
//				else   //左边数不存在
//				{
//					if (x.count(nums[i] + 1)!= 0) //左边数不存在,右边数存在
//					{
//						int newsequence = x[nums[i] + 1] + 1;
//						x[nums[i] + x[nums[i]+1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //左右都不存在
//						x[nums[i]] = 1;
//				}
//			}
//		}
//		return rst;
//	}
//};


//长度最小的子数组
//class Solution {
//public:
//	int minSubArrayLen(int s, vector<int>& nums) {
//		int left = 0, right = 0, sumlen = 0, rst = INT_MAX;
//		if (nums.size() == 0) return 0;
//		bool stopflag = false;
//		while (true)
//		{
//			  //当前和
//			if (sumlen>s && right!=nums.size())  //超过s,左滑窗右移动
//			{
//				rst = min(rst, right - left);
//				sumlen -= nums[left];
//				left++;
//			}
//			else if (sumlen == s && right != nums.size() )
//			{
//				rst = min(rst, right - left);
//				sumlen += nums[right];
//				right++;
//			}
//			else if (sumlen<s && right != nums.size())
//			{
//				sumlen += nums[right];
//				right++;
//			}
//			else  //右滑窗指到了边界
//			{
//				if (sumlen < s)
//				{
//					break;
//				}
//				else
//					if (sumlen == 2) rst = min(rst, right - left);
//					else
//				{
//					rst = min(rst, right - left);
//					sumlen -= nums[left];
//					left++;
//					continue;
//				}
//					
//			}
//		}
//		return rst == INT_MAX ? 0:rst;
//	}
//};
//
//class Solution {
//public:
//	bool increasingTriplet(vector<int>& nums) {
//		if (nums.size() <= 2) return false; int minnum = nums[0]; int nextmin =INT_MAX;
//		for (int i = 1; i < nums.size(); i++)
//		{
//			if (nums[i] < minnum){
//				nextmin = minnum;
//				minnum = nums[i];
//			}
//			else if (nums[i]<=nextmin)
//			{
//				nextmin = nums[i];
//			}
//			else  //nums[i]>nextmin
//			{
//				if (nextmin!=minnum)
//				return true;
//				else {
//					nextmin = nums[i];
//				}
//			}
//		}
//		return false;
//	}
//};
//
//class Solution {
//public:
//	void rotate(vector<vector<int>>& matrix) {
//		int roundtimes = matrix.size() / 2;
//		for (int i = 0; i < roundtimes;i++)
//		{
//			int maxvalue = matrix.size() - 1 - i;
//			for (int j = i; j < matrix.size();j++)  // [i][j]
//			{
//				int roundindex = matrix.size() - 1 - 2 * i;
//				int temp = matrix[i][j + roundindex];  //每一行移位的数量是 matrix.size()-1-2*i 
//				matrix[i][j + roundindex] = matrix[i][j];
//				matrix[i][j] = matrix[i + roundindex][j];
//				matrix[i + roundindex][j] = matrix[i + roundindex][j + roundindex];
//				matrix[i + roundindex][j + roundindex] = temp;
//			}
//		}
//	}
//};

//class Solution {
//public:
//	bool isPalindrome(string s) {
//		if (s.size() == 0) return true;
//		int head = -1, tail = s.size();
//		while (head < tail)
//		{
//			do 
//			{
//				head++; if (head >= s.size()) return true;
//				
//			} while (!(s[head] >= 'a'&&s[head] <= 'z' || s[head] >= 'A'&&s[head] <= 'Z'||s[head]>='0'&&s[head]<='9')); //head是字母时
//			do
//			{
//				tail--; if (tail<=0) return true;
//			} while (!(s[tail] >= 'a'&&s[tail] <= 'z' || s[tail] >= 'A'&&s[tail] <= 'Z' || s[tail] >= '0'&&s[tail] <= '9'));
//			if (s[head] == s[tail])
//				continue;
//			if (s[head] >= '0'&&s[head] <= '9') return false;
//			if (s[tail] >= '0'&&s[tail] <= '9') return false;
//			if (abs(s[head] - s[tail]) == 32) return true;
//			else return false;
//		}
//		return true;
//	}
//};

//class Solution {
//public:
//	int countsum(int n)
//	{
//		int total = 0;
//		while (n != 0)
//		{
//			total += pow(n % 10, 2);
//			n = n / 10;
//		}
//		return total;
//	}
//	bool isHappy(int n) {
//		set<int> s;
//		int current = n % 10;
//		while (s.find(n)==s.end())  //set中没有当前的数
//		{
//			s.insert(n);
//			if (countsum(n) == 1) return true;
//			else
//				n = countsum(n);
//		}
//		return false;
//	}
//};

//
//class Solution {
//public:
//	int findPeakElement(vector<int>& nums) {  //二分法，当二分法的左右指针相连续时，那么他的中间节点是左边的数
//		int n = nums.size();
//		if (n == 1) return 0;
//		if (n == 2) if (nums[1] > nums[0]) return 1; else return 0;
//		if (nums[n / 2] > nums[n / 2 + 1] && nums[n / 2] > nums[n / 2 - 1])
//			return n / 2;
//		if (nums[n / 2] < nums[n / 2 + 1])  //在右边寻找
//		{
//			for (int i = n / 2 + 1; i < n - 1; i++)
//			{
//				if (nums[i + 1] < nums[i]) return i;
//			}
//			return n - 1;
//		}
//
//		for (int i = n / 2 - 1; i >= 1; i--)
//		{
//			if (nums[i - 1] < nums[i]) return i;
//		}
//		return 0;
//	}
//};

//class Solution {
//public:
//	uint32_t reverseBits(uint32_t n) {
//		uint32_t ans = 0;
//		//进制的本质
//		int i = 32;
//		while (i--)
//		{
//			ans <<= 1;
//			ans += n & 1;
//			n >>= 1;
//		}
//		return ans;
//	}
//};

//struct Point {
//	int x;
//	int y;
//	Point() : x(0), y(0) {}
//	Point(int a, int b) : x(a), y(b) {}
//	
//};
//
//class Solution {
//public:
//	int maxPoints(vector<Point>& points) {    //每一次遍历一个点，通过一点加一个斜率确定唯一的直线，一次遍历找到所有直线通过该点的最大点数，遍历所有点数即得到了最优解。
//		int res = 0;
//		for (int i = 0; i < points.size(); ++i) {
//			map<pair<int, int>, int> m;
//			int duplicate = 1;
//			for (int j = i + 1; j < points.size(); ++j) {
//				if (points[i].x == points[j].x && points[i].y == points[j].y) {   //如果和下一个点是同一个点，如果是同一个点的话要记录，也算在一条直线上。
//					++duplicate; continue;
//				}
//				int dx = points[j].x - points[i].x;
//				int dy = points[j].y - points[i].y;
//				int d = gcd(dx, dy);
//				++m[{dx / d, dy / d}];
//			}
//			res = max(res, duplicate);
//			for (auto it = m.begin(); it != m.end(); ++it) {
//				res = max(res, it->second + duplicate);
//			}
//		}
//		return res;
//	}
//	int gcd(int a, int b) {
//		return (b == 0) ? a : gcd(b, a % b); //求最大公约数的算法
//	}
//};

//class Solution {
//public:
//	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
//		queue<string> q;
//		map<string, int> m1; //储存容器中的字符串便于查找是否存在
//		map<string, int> re; //储存结果
//		int n = wordList.size();
//		for (int i = 0; i < n; i++)
//			m1[wordList[i]] = 1;
//		re[beginWord] = 1;
//		q.push(beginWord);
//		while ((!q.empty()) && m1.size())
//		{
//			string now = q.front();
//			q.pop();
//			int num = re[now];
//			int llen = now.size();
//			for (int i = 0; i < llen; i++)
//			{
//				string temp = now;
//				for (char c = 'a'; c <= 'z'; c++)
//				{
//					if (temp[i] == c)
//						continue;
//					else
//						temp[i] = c;
//					if (m1.find(temp) != m1.end())
//					{
//						if (temp == endWord)
//							return num + 1;
//						q.push(temp);
//						re[temp] = num + 1;
//						m1.erase(temp);
//					}
//				}
//			}
//		}
//		return 0;
//	}
//};
//class Solution {
//public:
//	void reverseString(vector<char>& s) {
//		int j = s.size() - 1;
//		int i = 0;
//		while (i<j)
//		{
//			char x = s[i];
//			s[i] = s[j];
//			s[j] = x;
//			i++; j--;
//		}
//	}
//};

//const int MAXN = 26;
//class Trie{
//public:
//	bool is_str;  //判断当前节点是否为一个完整的字符串
//	Trie *next[MAXN];
//	Trie()
//	{
//		is_str = NULL;
//		memset(next, 0, sizeof(next));
//	}
//
//	void insert(string word)
//	{
//		Trie *cur = this;
//		for (char w:word)
//		{
//			if (cur->next[w-'a']==NULL)
//			{
//				Trie *new_node = new Trie();
//				cur->next[w - 'a'] = new_node;
//			}
//			cur = cur->next[w - 'a'];
//		}
//		cur->is_str = true;
//	}
//	bool search(string word) {
//		Trie *cur = this;
//		for (char w : word){
//			if (cur != NULL)
//				cur = cur->next[w - 'a']; // 更新cur指针的指向，使其指向下一个结点
//		}
//		return (cur != NULL&&cur->is_str); // cur指针不为空且cur指针指向的结点为一个完整的字符串，则成功找到字符串
//	}
//
//	/** Returns if there is any word in the trie that starts with the given prefix. */
//	bool startsWith(string prefix) {
//		Trie *cur = this;
//		for (char w : prefix){
//			if (cur != NULL)
//				cur = cur->next[w - 'a'];
//		}
//		return (cur != NULL); // 相比search(),这里只需判断cur指针是否为空就行了
//	}
//};

//class Solution {
//public:
//	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
//		int len = beginWord.length();
//		map<string, vector<string>> allComboDict;
//		map<string, bool> visited;
//		for (int j = 0; j < wordList.size();j++)
//		{
//			string word = wordList[j];
//			for (int i = 0; i < len; i++)
//			{
//				string newword = word.substr(0, i) + "*" + word.substr(i + 1, len - (i + 1));
//				allComboDict[word].push_back(newword);
//			}
//			visited[word]= false;
//		}
//		//Queue for BFS
//		queue<pair<string, int>> Q;
//		pair<string, int> a (beginWord, 1);
//		Q.push(pair<string, int> (beginWord, 1));
//		//a visited map
//		
//		visited[beginWord] = true;
//		while (!Q.empty())   //BFS终止循环条件，队列为空
//		{
//			pair<string, int>node = Q.front();
//			Q.pop();
//			string currentword = node.first;  //节点的string
//			int level = node.second;   //节点的level
//			for (int i = 0; i < len; i++)
//			{
//				string intermediate = currentword.substr(0, i) + "*" + currentword.substr(i + 1, len - (i + 1));
//				map<string, vector<string>>::iterator it = allComboDict.begin();
//				while (it != allComboDict.end())
//				{
//					for (int j = 0; j < (it->second).size();j++)
//					{
//						string intermediate1 = it->second[j];
//						if (intermediate == intermediate1)  //找到了联通的字段
//						{
//							if (it->first == endWord)
//								return level + 1;
//							else
//								if (visited[it->first] == false)  //没访问过
//								{
//									visited[it->first] = true;
//									Q.push(pair<string,int>(it->first, level + 1));
//								}
//							continue;
//						}
//						
//					}
//					it++;
//				}
//			}
//		}
//		//BFS
//		return 0;
//	} 
//
//};

//class Solution {
//public:
//	bool isAnagram(string s, string t) {
//		sort(s.begin(), s.end());
//		sort(t.begin(), t.end());
//		return s == t ? true : false;
//	}
//};


//
//const int Maxn = 26;
//class Trie {
//public:
//	/** Initialize your data structure here. */
//	bool is_str;//标识当前节点是否是一个完整的字符串
//	Trie *next[Maxn];  //指向Trie的指针数组
//	Trie() {
//		is_str = NULL;
//		memset(next, 0, sizeof(next)); //初始化指针数组
//	}
//
//	/** Inserts a word into the trie. */
//	void insert(string word) {
//		Trie *cur = this; //首先指向当前节点
//		for (char w : word)
//		{
//			if (cur->next[w - 'a'] == NULL)  //当前节点没有指向当前字符w的分支，要新建节点
//			{
//				Trie *new_node = new Trie();
//				cur->next[w - 'a'] = new_node;
//			}
//			cur = cur->next[w - 'a'];//每一次都会遍历一个分支，节点指向下一个字符
//		}
//		cur->is_str = true;//指向最末尾的字符了，当前节点处是一个完整的单词
//	}
//
//	/** Returns if the word is in the trie. */
//	bool search(string word) {
//		Trie *cur = this;
//		for (char w : word)
//		{
//			if (cur->next[w - 'a'] == NULL)
//			{
//				return false;
//			}
//			cur = cur->next[w - 'a'];
//		}
//		return (cur->is_str);
//	}
//
//	/** Returns if there is any word in the trie that starts with the given prefix. */
//	bool startsWith(string prefix) {
//		Trie * cur = this;
//		for (char w : prefix)
//		{
//			if (cur->next[w - 'a'] == NULL)
//				return false;
//			cur = cur->next[w - 'a'];
//		}
//		return (cur != NULL);
//	}
//};

//class Solution {
//public:
//	int titleToNumber(string s) {
//		int len = s.length(); int rst = 0;
//		for (int i = 0; i < s.length();i++)
//		{
//			char a= s[i];
//			switch (a)
//			{
//			case 'A': rst += 1 * pow(26, len - i - 1); break;
//			case 'B': rst += 2 * pow(26, len - i - 1); break;
//			case 'C': rst += 3 * pow(26, len - i - 1); break;
//			case 'D': rst += 4 * pow(26, len - i - 1); break;
//			case 'E': rst += 5 * pow(26, len - i - 1); break;
//			case 'F': rst += 6 * pow(26, len - i - 1); break;
//			case 'G': rst += 7 * pow(26, len - i - 1); break;
//			case 'H': rst += 8 * pow(26, len - i - 1); break;
//			case 'I': rst += 9 * pow(26, len - i - 1); break;
//			case 'J': rst += 10 * pow(26, len - i - 1); break;
//			case 'K': rst += 11 * pow(26, len - i - 1); break;
//			case 'L': rst += 12 * pow(26, len - i - 1); break;
//			case 'M': rst += 13 * pow(26, len - i - 1); break;
//			case 'N': rst += 14 * pow(26, len - i - 1); break;
//			case 'O': rst += 15 * pow(26, len - i - 1); break;
//			case 'P': rst += 16 * pow(26, len - i - 1); break;
//			case 'Q': rst += 17 * pow(26, len - i - 1); break;
//			case 'R': rst += 18 * pow(26, len - i - 1); break;
//			case 'S': rst += 19 * pow(26, len - i - 1); break;
//			case 'T': rst += 20 * pow(26, len - i - 1); break;
//			case 'U': rst += 21 * pow(26, len - i - 1); break;
//			case 'V': rst += 22 * pow(26, len - i - 1); break;
//			case 'W': rst += 23 * pow(26, len - i - 1); break;
//			case 'X': rst += 24 * pow(26, len - i - 1); break;
//			case 'Y': rst += 25 * pow(26, len - i - 1); break;
//			case 'Z': rst += 26 * pow(26, len - i - 1); break;
//			default:
//				break;
//			}
//		}
//		return rst;
//	}
//};

//struct BSTNode{
//	int val;  //当前值
//	int count;  //在nums中比val小的数的个数
//	BSTNode *left; //左子树指针
//	BSTNode *right; // 右子树指针
//	BSTNode(int x) :val(x), left(NULL), right(NULL), count(0){};//需要维护一个count，因为遍历时不是
//	//会将每个都遍历，所以要记录每一个节点的count值
//};
//
//void BST_insert(BSTNode *node, BSTNode *insert_node, int &count_small)
//{
//	//从node节点开始插入新的insert_node节点
//	if (node->val >= insert_node->val)    //新节点小于等于旧节点
//	{
//		node->count++;  //比当前节点小的节点个数要自增
//		if (node->left)   //新节点插入到左边
//			BST_insert(node->left, insert_node, count_small);
//		else
//			node->left = insert_node;
//		//插入的节点比当前节点要小，
//	}
//	else
//	{
//		count_small += node->count + 1;
//		if (node->right)  //当前节点的右子树不为空
//			BST_insert(node->right, insert_node, count_small);
//		else
//			//右子树是空
//			node->right = insert_node;
//	}
//}
//
//class Solution{
//public:
//	vector<int> countSmaller(vector<int> &nums)
//	{
//		int n = nums.size();
//		if (n == 0) return{}; //返回空的时候返回的是一个数组，用大括号表示
//		vector<int>  count;  // 保存最后结果的vector
//		count.push_back(0); // 最后一个元素值为0
//		BSTNode *node = new BSTNode(nums[n - 1]);  //树有n-1个节点,node是val为nums[n-1]的节点
//		int count_small;
//		for (int i = 1; i < n; i++)
//		{
//			count_small = 0; //每一次都从nums[n-1]的节点开始插入新的节点
//			BST_insert(node, new BSTNode(nums[n - i - 1]), count_small);//插入nums[n-i-1]个元素到二叉树中
//			count.push_back(count_small);
//		}
//		delete node;
//		reverse(count.begin(), count.end());
//		return count;
//	}
//};

//class Solution {
//public:
//	int findDuplicate(vector<int>& nums) {
//		sort(nums.begin(), nums.end());
//		for (int i = 0; i < nums.size()-1;i++)
//		{
//			if (nums[i]==nums[i+1])
//			{
//				return nums[i];
//			}
//		}
//		return -1;
//	}
//};

//class Solution {
//public:
//	//慢快指针的基本思想就是两个指针之差始终为慢指针所走的路程，两指针相遇以后，慢指针一定走了整数倍的环长度
//	// 因为初始走了m步，m是从起点到环的距离，所以此时慢指针再走m步就到了环的起点
//	// 此时让快指针从头走，走m步后也到环起点，两指针相遇，环起点也就是重复的数。
//	int findDuplicate(vector<int>& nums) {
//		int fast = nums[0];
//		int slow = nums[0];  //把nums[0]当做起始位置，再指针相遇时要回到nums[0]
//		do 
//		{
//			slow = nums[slow];  //走一步
//			fast = nums[nums[fast]]; //向前走两步
//		} while (fast!=slow);
//
//		// 相遇后快指针回到起点
//		fast = nums[0];
//		while (fast != slow)  //如果已经相遇，那么直接返回，不能用do while
//		{
//			fast = nums[fast];
//			slow = nums[slow];
//		} 
//
//		return fast;
//	}
//};
//class Solution {
//public:
//	int numIslands(vector<vector<char>>& grid) {
//		if (grid.size()==0)
//			return 0;
//		vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
//		int x = grid.size(), y = grid[0].size(); int res = 0;
//		for (int i = 0; i < x;i++)
//		{
//			for (int j = 0; j < y;j++)
//			{
//				if (grid[i][j]=='1')
//				{
//					res++; BFS(grid, i, j, visited);
//				}
//			}
//		}
//		return res;
//	}
//
//	void BFS(vector<vector<char>>& grid, int xx, int yy, vector<vector<bool>>& visited)
//	{
//		int x = grid.size(); int y = grid[0].size();
//		//x是第一组vector y是第二组vector的index
//		if (xx < 0 || xx >= x || yy < 0 || yy >= y)  //访问越界，返回
//			return;
//		else
//			if (visited[xx][yy] == true || grid[xx][yy] == '0')  //不符合条件
//				return;
//		  else
//		{
//			grid[xx][yy] = '0';
//			visited[xx][yy] = true;//设置为访问过
//			BFS(grid, xx + 1, yy, visited);
//			BFS(grid, xx - 1, yy, visited);
//			BFS(grid, xx , yy + 1, visited);
//			BFS(grid, xx , yy - 1, visited);
//		}
//	}
//};
//class Solution {
//public:  //三向切分看不懂，所以进阶不了。主要就是排序后注意放法 避免重复的数
//	void wiggleSort(vector<int>& nums) {
//		vector<int> copyed(nums);
//		if (nums.size() == 0) return ;
//		sort(copyed.begin(), copyed.end());
//		int len = nums.size()-1;
//		for (int i = 1; i < nums.size();i=i+2)
//	           //奇数放大数 
//				nums[i] = copyed[len--];
//		
//		for (int i = 0; i < nums.size(); i=i+2)
//			 //偶数放小数 
//			 nums[i] = copyed[len--];
//	}
//
//};
//

//class Solution {
//public:
//	bool increasingTriplet(vector<int>& nums) {
//		if (nums.size() < 3) return false;
//		int firstmin = INT_MAX, secondmin = INT_MAX;
//		for (int i = 0; i < nums.size(); i++)
//		{
//			if (nums[i]<firstmin) 
//				firstmin = nums[i];
//			else 
//				if (nums[i] = firstmin) continue;
//				else 
//					if (nums[i] < secondmin) secondmin = nums[i];
//					else if (nums[i] > secondmin) return true;
//			
//		
//		}
//		return false;
//	}
//};


//// 今天就做了两个二叉树，很快就学会了，完成了打卡，气死强哥，就是不给你买奶茶 haha~
// struct TreeNode {
//     int val;
//     TreeNode *left;
//     TreeNode *right;
//     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// };
// //对搜素二叉树中序遍历，实际上就是遍历得到了依次从小到大排列的节点值
//class Solution {
//public:
//	vector<int>result;
//	int  kthSmallest(TreeNode* root,int k) {
//		if (root == NULL) return 0;
//		if (root->left != NULL)  //左边有访问左边
//			kthSmallest(root->left,1);
//		//左边没有，到它自己了，所以返回它的值后访问右边
//		result.push_back(root->val);
//		if (root->right != NULL) kthSmallest(root->right,1);
//		return result[k-1];
//	}
//};




////主要是这个递归的思路得先想到，要掌握大的思路不要纠结细枝末节，要不就绕不出来了。得多复习几次。
//class Solution {
//public:
//	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
//		if (root == p || root == q) return root;//找到了指定节点直接返回
//		if (root == NULL) return NULL; //搜寻到最后都没有搜寻到，说明没找到，返回NULL
//		//没有找到就要开始搜寻了,搜索左边
//		TreeNode *left = lowestCommonAncestor(root->left, p, q);
//		TreeNode *right = lowestCommonAncestor(root->right, p, q);
//
//		//中心原理
//		if (left == NULL&&right ==NULL) //都没搜索到 
//			return NULL;
//		if (left != NULL&&right == NULL)  //左有右没有
//			return left;
//		if (left == NULL&&right != NULL)
//			return right;
//		else return root; //左有都有
//	}
//};
//
//struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//
//};
//
//class Solution {
//public:
//	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
//		if (root->val == p->val || root->val == q->val) return root;
//		else
//			if (root->val > p->val&&root->val > q->val) //都小于当前节点，肯定在右边
//			{
//				return lowestCommonAncestor(root->left, p, q);
//			}
//			else if (root->val < p->val&&root->val < q->val)
//			{
//				return lowestCommonAncestor(root->right, p, q);
//			}
//			else
//				return root;
//	}
//};


//动态规划，把整个问题分为小的问题。dp[i]表示包含第i位为子序列的情况下最长的上升序列
//所以下一状态就是 dp[i+1] 要更新，nums[i+1]就要依次和nums[i],nums[i-1]...nums[0]比较来更新dp[i+1]
//做动态规划的时候，首先要把逻辑想清楚，再写代码。
//class Solution {
//public:
//	int lengthOfLIS(vector<int>& nums) {
//		int rst = 1; int median = 1;
//		if (nums.size() == 0) return 0;
//		vector<int> dp(nums.size(), 1);
//		dp[0] = 1;
//		for (int i = 1; i < nums.size();i++)
//		{
//			for (int j = i - 1; j >= 0;j--)
//			{
//				//遍历i-2到0,找到dp[i]
//				if (nums[i] < nums[j]) median = 1;
//				else 
//					if (nums[i]>nums[j]) 
//				{
//					median = dp[j] + 1;
//				}
//				else
//				{   //两数相等
//					median = dp[j];
//				}
//				dp[i] = max(dp[i], median);
//			}
//			rst = max(rst, dp[i]);
//		}
//		return rst;
//
//	}
//};

// 暴力解法，遍历子序列，并对每一个子序列进行判断是否满足条件，满足就判断size是否变大，记录最新的size
// 但是有一组没有通过，超时了
//class Solution {
//public:
//	int longestSubstring(string s, int k) {
//		int rst = 0; int len = s.size();
//		if (s.size() == 0) return 0;
//		//map<int,int> times;
//		for (int i = 0; i < len; i++)
//		{
//			map<int, int> times;
//			times[s[i] - 'a']++; //记录当前字符个数
//			for (int j = i + 1; j < len; j++)
//			{
//				//遍历从i到结尾，相当于遍历了所有的子串
//				times[s[j] - 'a']++;
//				if (iscorrect(times, k) == true)  //当前子串满足条件，记录大小;
//				{
//					rst = max(rst, j - i + 1);
//				}
//			}
//		}
//		if (k == 1)
//			return rst > 1 ? rst : 1;
//		else
//			return rst;
//	}
//	bool inline iscorrect(map<int, int> times, int k)
//	{
//		int lenth = times.size();
//		map<int, int>::iterator it = times.begin();
//		while (it != times.end())
//		{
//			if (it->second < k) return false;
//			it++;
//		}
//		return true;
//	}
//};

//这个解法实际上也就是上面暴力的优化，我写的判断是否满足条件是对map进行访问，判断每个元素的出现次数
//这个解法对这一个函数进行优化，就用一个int的掩膜，对每一位进行1或0的操作
//掩膜为0的时候，也就是满足条件的时候，就避免了遍历
//第二个是外层循环，只要i+k>n，就不用了遍历了，这是第二个优化
//第三个优化就是i = i+max_idx,这个我想了半天也没弄明白。
// 因为我看了很久还没完全明白，这个题不想自己写了。
//class Solution {
//public:
//	int longestSubstring(string s, int k) {
//		int res = 0, i = 0, n = s.size();
//		while (i + k <= n) {//关键
//			int m[26] = { 0 }, mask = 0, max_idx = i;
//			for (int j = i; j < n; ++j) {
//				int t = s[j] - 'a';
//				++m[t];
//				if (m[t] < k) mask |= (1 << t);//关键
//				else mask &= (~(1 << t));  //关键
//				if (mask == 0) {
//					res = max(res, j - i + 1);
//					max_idx = j;
//				}
//			}
//			i = i + max_idx; //没看懂，但是i++也能通过
//		}
//		return res;
//	}
//};

//自己随便写的，其他更好方法就学习一下就不写了
//class Solution {
//public:
//	int getSum(int a, int b) {
//		if (b == 0) return a;
//		else
//			if (b>0)
//			{
//				while ((b--)!=0)
//				{
//					a++;
//				}
//				return a;
//			}
//			else
//			{
//				if (b < 0)
//				{
//					while ((b++) != 0)
//					{
//						a--;
//					}
//					return a;
//				}
//			}
//		return a;
//	}
////};
//
//class Solution {
//public:
//	int majorityElement(vector<int>& nums) {
//		int counttimes = 1;
//		if (nums.size() == 1) return nums[0];
//		int majority = nums[0];
//		for (int i = 1; i < nums.size();i++)
//		{
//			if (nums[i]==majority)
//				counttimes++;
//			else
//			{
//				if (counttimes ==0)
//				{
//					majority = nums[i];
//					counttimes++;
//				}
//				else
//				{//不为majority，countimes也不为0
//					counttimes--;
//				}
//			}
//			
//		}
//		return majority;
//	}
//};

//代码很乱，可以简化就不简化了，今天主要是掌握了摩尔投票法的思路
//class Solution {
//public:
//	vector<int> majorityElement(vector<int>& nums) {
//		int countx = 0, county = 0;
//		int majorityx = 0, majorityy = 0; vector<int>rst;
//		if (nums.size() == 0) return rst;
//		if (nums.size()==1)  
//		{
//			rst.push_back(nums[0]);
//			return rst;
//		}
//		else if (nums.size()==2)
//		{
//			if (nums[1] != nums[0])
//			{
//				rst.push_back(nums[0]);
//				rst.push_back(nums[1]);
//			}
//			else
//				rst.push_back(nums[0]);
//			return rst;
//		}
//		for (int i = 0; i < nums.size();i++)
//		{
//			
//			if (countx==0&&nums[i]!=majorityy)
//			{
//				majorityx = nums[i];
//				countx++;
//			}
//			else
//				if (nums[i] == majorityx)
//				{
//					countx++;
//				}
//				else//都不为0时
//					if (county == 0)
//					{
//						majorityy = nums[i];
//						county++;
//					}
//					else if (nums[i] == majorityy)
//					{
//						county++;
//					}
//					else
//					{
//						countx--; county--;
//					}
//		}
//		countx = 0; county = 0;
//		for (int i = 0; i < nums.size();i++)
//		{
//			if (majorityx==nums[i])
//			{
//				countx++;
//			}
//			else if (majorityy == nums[i])
//			{
//				county++;
//			}
//
//		}
//		if (countx>nums.size()/3)
//		{
//			rst.push_back(majorityx);
//		}
//		if (county > nums.size() / 3)
//		{
//			rst.push_back(majorityy);
//		}
//		return rst;
//	}
//};

//int calculate(string s) {
//	int res = 0, d = 0;
//	char sign = '+';
//	stack<int> nums;
//	for (int i = 0; i < s.size(); ++i) {
//		if (s[i] >= '0') {//取到的是数字
//			d = d * 10 - '0' + s[i];//进位(先减法),取到一位数字对数字还原
//		}
//		if ((s[i] < '0' && s[i] != ' ') || i == s.size() - 1) {  //如果当前是符号位或是最后一位了，就会判断上一次的符号位sign
//			if (sign == '+') {
//				nums.push(d);
//			}
//			else if (sign == '-') {
//				nums.push(-d);
//			}
//			else if (sign == '*' || sign == '/') {
//				int tmp = sign == '*' ? nums.top() * d : nums.top() / d;
//				nums.pop();  //取出了最后一个数，利用stack后进先出的性质
//				nums.push(tmp);
//			}
//			sign = s[i]; //保存当前符号
//			d = 0;
//		}
//	}
//	for (; !nums.empty(); nums.pop()) {
//		res += nums.top();
//	}
//	return res;
//}


//堆栈的题目，因为遍历的顺序，如果都是加法的话都push到stack中就没有先后顺序，关键是有了*/
//它们的优先级别要高一些，所以当遍历到*/时要将之前push到stack中的数字取出来，是后放进去的先取
//所以要用一个stack来去保存遍历到的数字。 遍历到该乘法除法运算的时候，就把stack中的数字pop出来
//运算了以后将乘法除法的结果push到stack中，最后stack元素相加即可。
//有一些小的细节比如计算currentnum时要先-’0‘再+s[i],反过来有些数字会越界！
//class Solution {
//public:
//	int calculate(string s) {
//		int len = s.size();
//		stack<int> nums;
//		int currentnum = 0;
//		int  rst = 0;
//		char presign = '+'; //定义要计算时上一次的符号
//		for (int i = 0; i < len;i++)
//		{
//			if (s[i] >= '0'&&s[i] <= '9') //取到的元素是数字
//				currentnum = currentnum * 10 - '0' + s[i];//先加法的话会出界！采坑。
//			
//			if (s[i] < '0'&&s[i] != ' '||i==len-1)  //取到加减乘除表示当前的数字取完了，可以push到stack
//			{
//				if (presign == '+')
//				{
//					presign = s[i];
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '-')
//				{
//					presign = s[i];
//					nums.push(-currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '*')
//				{
//					presign = s[i];
//					currentnum = nums.top()*currentnum;
//					nums.pop();
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '/')
//				{
//					presign = s[i];
//					currentnum = nums.top()/currentnum;
//					nums.pop();
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//					
//			}
//		}
//
//		while (!nums.empty())
//		{
//			rst += nums.top();
//			nums.pop();
//		}
//		return rst;
//	}
//};

//每次快排结束后基准数的位置就代表着他是数组中第k大的数。
//这是快排序从大到小来排，利用快排序排完了以后返回的元素排的元素左边是大于他的 右边是小于他的。
//那这个数的下标是index的话 他不就是第index加1大的数？ 每次排完确定了一个第INDEX大的数。
//然后和k比较，可以确定新的数实在左边序列还是右边序列，再继续快排就好了。
//class Solution {
//public:
//	int findKthLargest(vector<int>& nums, int k) {
//		int low = 0, high = nums.size() - 1, mid = 0;
//		while (low <= high)
//		{
//			mid = partation(nums, low, high);  //第mid+1大的数为nums[mid]
//			if (mid + 1 == k) return nums[mid];
//			else if (mid + 1 < k)  //在右边
//			{
//				low = mid + 1; 
//			}
//			else
//			{
//				high = mid - 1;
//			}
//
//		}
//		return 1;
//	}
//
//	int partation(vector<int> &nums, int low, int high) //low和high代表搜寻的范围
//	{
//		int left = low + 1, right = high;
//		int bound = nums[low];//参考比对数: bound
//		while (left <= right)
//		{
//			while (left < high && nums[left] > bound) left++;  //左指针向右，找到第一个比bound小的数后停下
//			while (nums[right] < bound &&right>low) right--;
//			if (left < right) swap(nums[left++], nums[right--]);
//			else break;
//		}
//		if (right < left)
//		{
//			swap(nums[low], nums[right]); return right;
//		}
//		else
//		{
//			swap(nums[low], nums[left]);
//			return left;  //返回的数也就是当前比对参数的坐标，把这个数第left+1大的数
//		}
//	}
//};


// Employee info
//class Employee {
//public:
//// It's the unique ID of each node.
//// unique id of this employee
//int id;
//// the importance value of this employee
//int importance;
//// the id of direct subordinates
//vector<int> subordinates;
//Employee(int id, int importance, vector<int> subordinates)
//{
//	this->id = id;
//	this->importance = importance;
//	this->subordinates = subordinates;
//}
//};
//
//class Solution {
//public:
//	int getImportance(vector<Employee*> employees, int id) {
//		map<int, Employee*> employeemap ;//建立一个员工id和相应的它的数据结构的map映射关系
//		map <int, bool> visited;  //建立一个visited
//		queue<int> BFSqueue;
//		for (int i = 0; i < employees.size(); i++)
//		{
//			employeemap[employees[i]->id] = employees[i];
//			visited[employees[i]->id] = false;
//		}
//
//		//BFS搜索,从参数所给的id开始
//		BFSqueue.push(id); int importance=0;
//		while (!BFSqueue.empty())
//		{
//			int curid = BFSqueue.front();
//			visited[curid] = true; 
//			BFSqueue.pop(); importance += employeemap[curid]->importance;
//			for (int i = 0; i < (employeemap[curid]->subordinates).size(); i++)
//			{
//				//访问的是当前员工的下属的第i个下属编号
//				if (visited[(employeemap[curid]->subordinates)[i]] == false)
//				{
//					BFSqueue.push((employeemap[curid]->subordinates)[i]);
//				}
//			}
//		}
//		return importance;
//	}
//};
//class orange
//{
//public: 
//	int x;
//	int y;
//	int minute;
//	orange(int a, int b,int min):x(a), y(b),minute(min){}
//};
//class Solution {
//public:
//	int orangesRotting(vector<vector<int>>& grid) {
//		int row = grid.size(); int rst = 0;
//		int col = grid[0].size(); int counttotal = 0;
//		queue<orange> BFSqueue; int countrot = 0;
//		for (int i = 0; i < row; i++)
//		{
//			for (int j = 0; j < col; j++)
//			{
//				if (grid[i][j] == 2) //如果是腐烂的橘子就加入到队列
//				{
//					BFSqueue.push(orange(i, j,0));  //将橘子信息加入到队列中，方便查找
//					countrot++;
//				}
//				if (grid[i][j] == 1)
//				{
//					counttotal++;
//				}
//			}
//		}
//		counttotal += countrot;
//		
//		while (!BFSqueue.empty())
//		{
//			orange curcorange = BFSqueue.front();
//			int x = curcorange.x; int y = curcorange.y; int level = curcorange.minute;
//			BFSqueue.pop();
//			if (x+1 >= 0 && x + 1 <= row - 1 && y >= 0 && y <= col - 1 && grid[x + 1][y] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x + 1, y, level + 1));
//				rst = max(level + 1, rst);
//				grid[x + 1][y] = 2;
//			}
//			if (x - 1 >= 0 && x - 1 <= row - 1 && y >= 0 && y <= col - 1 && grid[x - 1][y] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x - 1, y, level + 1));
//				rst = max(level + 1, rst);
//				grid[x - 1][y] = 2;
//			}
//			if (x >= 0 && x <= row - 1 && y + 1 >= 0 && y + 1 <= col - 1 && grid[x ][y + 1] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x , y + 1, level + 1));
//				rst = max(level + 1, rst);
//				grid[x][y + 1] = 2;
//			}
//			if (x >= 0 && x <= row - 1 && y - 1 >= 0 && y - 1 <= col - 1 && grid[x ][y - 1] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x , y - 1, level + 1));
//				rst = max(level + 1, rst);
//				grid[x][y - 1] = 2;
//			}
//
//		}
//		if (countrot < counttotal) return -1;
//		else
//			return rst;
//	} 
//};

//struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	
//};
//
//class Solution {
//public:
//	int pax = 0; int pay = 0; int levelx = 0; int levely = 0;
//	bool findx = false;	bool findy = false;
//	bool isCousins(TreeNode* root, int x, int y) {
//		findxy(root, root, x, y, 0);
//		return (levelx == levely&&pax != pay) ? true : false;
//
//	}
//	void findxy(TreeNode* curnode, TreeNode* previousNode, int x, int y, int level)
//	{
//		if (curnode == NULL) return;
//		if (curnode->val ==x)
//		{
//			cout << "找到了x level=" << level << endl;
//			levelx = level; pax = previousNode->val; findx = true;
//		}
//		if (curnode->val == y)
//		{
//			cout << "找到了y level=" << level << endl;
//			levelx = level; pay = previousNode->val; findy = true;
//		}
//		if (curnode->left != NULL && !(findx == true && findy == true))
//		{
//			cout << "寻找" << curnode->val + "的左节点  level=" << level << endl;
//			findxy(curnode->left, curnode, x, y,level+1);
//			
//		}
//		if (curnode->right != NULL && !(findx == true && findy == true))
//		{
//			cout << "寻找" << curnode->val + "的右节点  level=" << level << endl;
//			findxy(curnode->right, curnode, x, y, level + 1);
//		}
//	}
//};
//
//class Solution {
//	vector<vector<int>> rst;
//public:
//	vector<vector<int>> pathSum(TreeNode* root, int sum) {
//		vector<int>curpath; 
//		aimedpath(root, sum, curpath, 0);
//		return rst;
//	}
//
//	void aimedpath(TreeNode*root,int sum,vector<int> curpath,int cursum)
//	{
//
//		if (root == NULL) return;
//		cursum += root->val;
//		if (cursum == sum&&root->left==NULL&&root->right==NULL) {
//			curpath.push_back(root->val);  //当前节点下去就不用考虑了
//			rst.push_back(curpath);
//			return;
//		}
//			curpath.push_back(root->val);
//			aimedpath(root->left, sum, curpath, cursum);
//			aimedpath(root->right, sum, curpath, cursum);
//	}
//};

//struct TreeNode {
//	     int val;
//	     TreeNode *left;
//	     TreeNode *right;
//	     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	 };
//
////二叉搜索树的中序遍历
//class Solution {
//public:
//	int current; int previous; int rst = true;
//	bool isValidBST(TreeNode* root) {
//		current = root->val;
//		middleBFS(root);
//		return rst;
//	}
//
//	bool middleBFS(TreeNode* root)
//	{
//		if (rst == false) return false;
//		if (root == NULL) return;
//		if (root->left != NULL) middleBFS(root->left);
//		previous = current;
//		current = root->val;
//		if (current < previous) rst = false;
//		if (root->right != NULL) middleBFS(root->right);
//		return true;
//	}
//};
//

//想学习贪心算法，从简单的开始，但这题太过于简单，就算不会贪心算法也知道题解啊
//class Solution {
//public:
//	bool lemonadeChange(vector<int>& bills) {
//		if (bills[0] != 5 || bills[1] == 20)//自己初始为0，所以第一回合后自己手中的钱应该为5，第二回合应为10
//			return false;
//		int n = 0;//5元的数目
//		int m = 0;//10元的数目
//		for (int i = 0; i < bills.size(); i++)
//		{
//			if (bills[i] == 5)
//			{
//				n++;
//			}
//			else if (bills[i] == 10)
//			{
//				if (n)
//				{
//					n--;
//					m++;
//				}
//				else return false;
//			}
//			else
//			{
//				if (m)
//				{
//					if (n)
//					{
//						m--;
//						n--;
//						continue;
//					}
//				}
//				if (n >= 3)
//				{
//					n -= 3;
//				}
//				else return false;
//			}
//		}
//		return true;
//	}
//};

//也是贪心算法，一下子就写出来了，感觉贪心算法更多的是一种思路。
//class Solution {
//public:
//	bool canJump(vector<int>& nums) {
//		if (nums.size() == 1) return true;;
//		int len = nums.size();
//		int curbest = len - 1;
//		for (int i = len-2; i>=1; i--)
//		{
//			if (nums[i] + i >= curbest)
//			{//当前位置就是最近的最好位置，更新
//				curbest = i;
//			}
//		}
//		return nums[0] >= curbest ? true:false;
//	}
//};
//
//

//回溯算法,对其进行优化就是记录每一个点的位置，记录这个点是可以通向终点的还是是思路，下次遍历的时候就不用走到头了
//class Solution {
//public:
//	 //0 未知，1 
//	int jump(vector<int>& nums) {
//		int step = 0; int maxpos = 0; int len = nums.size() - 1;
//		for (int i = 0; i < len;)
//		{
//			int best = 0; int ini = 0;
//			
//			for (int j = 1; j <= nums[i];j++)  //在第i步时往前走的步数
//			{
//				if (i + j == len) return step + 1;
//
//				if (j+nums[i+j]>ini)  //走j步好一些
//				{
//					best = j;  //更新为向前走j步
//					ini = j + nums[i + j];
//				}
//			}
//			i = i + best;
//			step++;
//		}
//		return step;
//	}
//
//	
//};


//只看懂了一点点 所以照着别人的写了一遍。。。
//class Solution {
//public:
//	int videoStitching(vector<vector<int>>& clips, int T) {
//		if (T == 0)
//			return 0;
//		int len = clips.size();
//		int l, r;
//		l = 0;
//		r = T;
//		int re = 0;
//		while (l<r)
//		{
//			int maxl, maxr;
//			int indl, indr;
//			maxl = maxr = -1;
//			indl = l;
//			indr = r;
//			for (int i = 0; i<len; ++i)
//			{
//				if (clips[i][0] <= l)
//				{
//					if ((clips[i][1] - l)>maxl)
//					{
//						maxl = (clips[i][1] - l);
//						indl = i;
//					}
//				}
//				if (clips[i][1] >= r)
//				{
//					if ((r - clips[i][0])>maxr)
//					{
//						maxr = (r - clips[i][0]);
//						indr = i;
//					}
//				}
//			}
//			if (maxl == -1 || maxr == -1)
//				return -1;
//			++re;
//			if ((r) <= (clips[indl][1]) || l >= clips[indr][0])
//				break;
//			++re;
//			l = clips[indl][1];
//			r = clips[indr][0];
//		}
//		return re;
//	}
//};
//
//class Solution {
//public:
//	int Max1 = INT_MIN;
//	int maxPathSum(TreeNode* root) {
//		if (root == NULL) return INT_MIN;
//
//		digui(root);
//
//
//		return Max1;
//	}
//
//	int digui(TreeNode* root){
//		if (root == NULL) return INT_MIN;
//
//		int Max2 = root->val;
//		int Max3 = root->val;
//		int path1 = digui(root->left);
//		int path2 = digui(root->right);
//
//		if (path1 > 0)  //path1表示左边路径
//			Max2 = max(Max2, root->val + path1);
//		if (path2 > 0)  //右边路径
//			Max2 = max(Max2, root->val + path2);
//		if (path1 > 0 && path2 > 0)  //
//			Max3 = root->val + path1 + path2;
//		Max1 = max(Max1, Max2);
//		Max1 = max(Max1, Max3);
//
//		cout << "节点" << root->val << "的最大和" << Max1 << endl;
//		return Max2;
//	}
//
//};

//
//class Solution {
//public:
//	bool isPossible(vector<int>& nums) {
//		map <int, int> countnum;	map <int, int> tailmap;
//		for (int i = 0; i <= nums.size()-1;i++)
//		{
//			countnum[nums[i]]++;
//			
//		}
//	//对于每一个序列，只用记住队尾的数列就行，因为是升序的
//		for (int i = 0; i <= nums.size()-1;i++)
//		{
//			
//			if (countnum[nums[i]]==0)  continue;//没有元素
//			else
//			{
//				//存在这个元素
//				if(tailmap[nums[i]-1] == 0) //队尾没有
//				{
//					if (countnum[nums[i] + 1] != 0 && countnum[nums[i] + 2] != 0)
//					{
//						//后面两个元素都有
//						tailmap[nums[i] + 2]++; countnum[nums[i]]--;
//						countnum[nums[i] + 1]--; countnum[nums[i] + 2]--;
//					}
//					else return false;
//				}
//				else
//				{
//					//队尾有这个元素，添加到队尾即可
//					tailmap[nums[i]]++; tailmap[nums[i]-1]--;
//					countnum[nums[i] ]--;
//				}
//
//			}
//		}
//		return true;
//	}
//};

//class Solution {
//public:
//	int candy(vector<int>& ratings) {
//		if (ratings.size() <= 1) return ratings.size();
//		int candies = 0; int up = 0; int old_slope = 0; int down = 0;
//		for (int i = 1; i <= ratings.size() - 1;i++)
//		{
//			//如果ratings[i]>ratings[i-1],newslop为1，等于为0；小于为-1;
//			int new_slope = (ratings[i] > ratings[i - 1]) ? 1 : (ratings[i] < ratings[i - 1] ? -1 : 0);
//			if ((old_slope>0&&new_slope==0)||(old_slope<0&& new_slope>=0))
//			{
//				candies += countc(up) + countc(down) + max(up, down); up = 0; down = 0;
//			}
//			if (new_slope > 0) up++;
//			if (new_slope < 0) down++;
//			if (new_slope == 0) candies++;
//			old_slope = new_slope;
//		
//		}
//		candies = countc(up) + countc(down) + max(up, down) + 1;
//		return candies;
//	}
//
//	int countc(int n)
//	{
//		return n*(n + 1) / 2;
//	}
//};

// struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	
//};
// class Solution {
// public:
//	 TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
//		 TreeNode *root = new TreeNode(0);
//		 return construct(0, nums.size() - 1, nums);
//	 }
//
//	 TreeNode* construct(int le, int ri, vector<int>& nums)
//	 {
//		 if (le<0 || ri>nums.size() - 1 || le > ri) { cout << "le:" << le << " ri:" << ri << "返回" << endl; return NULL; }
//		 if (le == ri) { TreeNode *node = new TreeNode(nums[le]); return node; }
//		 int maxnum = INT_MIN; int index = le;
//		 //第一个参数是插入的当前节点，第二个参数是开始位置，第三个参数是结束位置
//		 for (int i = le; i <= ri; i++)
//		 {
//			 if (nums[i] > maxnum)
//			 {
//				 index = i; maxnum = nums[i];
//			 }
//		 }
//		 cout << "le:" << le << " ri:" << ri << "Index:" << index << "建立了" << nums[index] << endl;
//		 TreeNode *node = new TreeNode(nums[index]);
//		 node->left = construct(le, index - 1, nums);
//		 node->right = construct(index + 1, ri, nums);
//		 return node;
//	 }
// };
//
//class Solution {
//public:
//	int uniquePaths(int m, int n) {
//		int left = m - 1, total = m + n - 2;
//		return zuhe(total, left);
//	}
//	int zuhe(int n,int m)
//	{
//		int rst = 1;
//		for (int i = m + 1; i <=n ; i++)
//		{
//			rst *= i;
//		}
//		for (int i = 2; i <= n-m; i++)
//		{
//			rst /= i;
//		}
//		return rst;
//	}
//};

//class Solution {
//public:
//	vector<string> rst;
//	vector<string> letterCombinations(string digits) {
//		vector<string> indexstring = { "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
//		if (digits == "") return rst;
//	
//		digui(digits, 0, "",indexstring);
//		return rst;
//	
//	}
//	void digui(string digits, int index,string custring,vector<string> indexstring )
//	{
//		if (index == digits.size())
//		{
//			rst.push_back(custring); return;
//		}
//		for (int i = 0; i < indexstring[digits[index]-'2'].size();i++)
//		{
//			string custring1;
//			custring1=custring + indexstring[digits[index] - '2'][i];
//			digui(digits, index + 1, custring1, indexstring);
//			
//		}
//	}
//};


//class Solution {
//public:
//	vector<int> spiralOrder(vector<vector<int>>& matrix) {
//		vector<int>rst;
//		int right  = matrix[0].size() - 1;
//		int down = matrix.size() - 1;
//		int left = 0;
//		int up = 0;
//		while (1)
//		{
//			//向右，从matrix[left]走到
//			for (int i = left; i <= right; i ++ ) rst.push_back(matrix[up][i]);
//			if (++up > down) return rst;
//			for (int i = up; i <= down; i++) rst.push_back(matrix[i][right]);
//			if (--right<left) return rst;
//			for (int i = right; i >= left; i--) rst.push_back(matrix[down][i]);
//			if (--down<up)return rst;
//			for (int i = down; i >=up; i--) rst.push_back(matrix[i][left]);
//			if (++left > right) return rst;
//		}
//		return rst;
//	}
////};
//class Solution {
//public:
//	vector<vector<int>> generateMatrix(int n) {
//		vector<vector<int>> rst(n, vector<int>(n)); int cur = 1;
//		if (n == 1){ rst[0][0] = 1; return rst; }
//		int left = 0; int right = n - 1; int down = n-1; int up = 0;
//		while (1)
//		{
//			for (int i = left; i <= right; i++) rst[up][i] = cur++;
//			up++; if (cur > n*n) break;
//			for (int i = up; i <= down; i++) rst[i][right] = cur++;
//			right--; if (cur > n*n) break;
//			for (int i = right; i >= left; i--) rst[down][i] = cur++;
//			down--; if (cur > n*n) break;
//			for (int i = down; i >= up; i--)
//				rst[i][left] = cur++;
//			left++; if (cur > n*n) break;
//		}
//		return rst;
//	}
//};
//
//class Solution {
//public:
//	int firstUniqChar(string s) {
//		map<char, int> coutt; int curindex = -1;
//		for (int i = 0; i < s.size(); i++)
//		{
//			coutt[s[i]]++;
//		}
//		for (int i = 0; i < s.size(); i++)
//		{
//			if (coutt[s[i]] == 1) return i;
//		}
//		return -1;
//	}
//};

//
//class Solution {
//public:
//	int evalRPN(vector<string>& tokens) {
//		stack<int> s; int rst = 0;
//		if (tokens.size() == 1) return atoi(tokens[0].c_str());
//		for (int i = 0; i < tokens.size();i++)
//		{
//			string str = tokens[i];
//			if (str == "+")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a + b;
//				s.push(rst); continue;
//			}
//			if (str == "-")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a - b;
//				s.push(rst); continue;
//			}
//			if (str == "*")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a*b;
//				s.push(rst); continue;
//			}
//			if (str == "/")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a / b;
//				s.push(rst); continue;
//			}
//			s.push(atoi(str.c_str()));
//		}
//		return rst;
//	}
//};

//class Solution {
//public:
//	bool flag = false;
//	bool wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size(); 
//		if (s.size() == 0 || wordDict.size() == 0) return false;
//			
//		digui(s, 0, len, wordDict);
//		return flag;
//	}
//	void digui(string s, int index, int len, vector<string> wordDict)
//	{
//		if (index == len) 
//			flag = true;
//		if (flag == true) return;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curstring = wordDict[i];
//			if (index+curstring.size()>len) continue;
//			else
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					digui(s, index + curstring.size(), len, wordDict);
//				}
//		}
//	}
//};


//class Solution {
//public:
//	bool flag = false;
//	
//	bool wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size();
//		if (s.size() == 0 || wordDict.size() == 0) return false;
//		vector<int> rem(s.size(), -1);
//		digui(s, 0, len, wordDict,rem);
//		return flag;
//	}
//	void digui(string s, int index, int len, vector<string> wordDict,vector<int> &rem)
//	{
//		if (index == len)
//			flag = true;
//		if (flag == true||rem[index]==0) return ;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curstring = wordDict[i];
//			if (index + curstring.size()>len) continue;
//			else
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					digui(s, index + curstring.size(), len, wordDict, rem);
//				}
//		}
//		if (flag == false) rem[index] = 0;
//	}
//};

//class Solution {
//public:
//	vector<string> wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size();
//		vector<string> rst; if (s.size() == 0 || wordDict.size() == 0) return rst;
//		vector<int> rem(s.size(), -1); bool thistime = false;
//		digui(s, 0, len, wordDict, rem, "", rst, thistime);
//		return rst;
//	
//	}
//	void digui(string s, int index, int len, vector<string> wordDict, vector<int> &rem,string curs,vector<string>& rst,bool &thistime)
//	{
//		
//		if (index == len) { rst.push_back(curs); thistime = true;  return; }
//		if (rem[index] == 0) 
//			return;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curs1 = curs;
//			string curstring = wordDict[i];
//			if (index + curstring.size()>len) continue;
//			else
//			{
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					if (index == 0) curs1 = curstring;
//					else
//						curs1 += " " + curstring;
//					digui(s, index + curstring.size(), len, wordDict, rem, curs1,rst,thistime);
//				}
//			}
//		}
//		if (thistime == false)
//			rem[index] = 0;
//		
//		
//	} 
//};

class Solution {
public:
	vector<vector<string>> rst;
	vector<vector<string>> partition(string s) {
		vector<string> cur;
		digui(0, cur, s);
		return rst;
	}

	void digui(int index, vector<string>cur,string s)
	{
		if (index >= s.size()) {
			rst.push_back(cur); return;
		}
		for (int i = 1; (int)(s.size())-i-index>=0;i++)
		{
			string sub = s.substr(index, i);
			if (isHui(sub) == true) {
				cur.push_back(sub);
				digui(index + i, cur, s);
				cur.pop_back();
			}
		}
	}

	bool isHui(string s)
	{
		int first = 0; int last = s.size()-1;
		while (first<last)
		{
			if (s[first] == s[last]) { first++; last--; continue; }
			else
				return false;
		}
		return true;
	}
};

int main(int argc, char *argv[])
{
	string s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
	vector<string> wordDict = { "a", "aa", "aaa", "aaaa", "aaaaa" };
	Solution t;
	t.partition("efe");
}
	















