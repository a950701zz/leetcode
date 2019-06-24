#include <vector>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <algorithm>    // std::max
#include "string"
#include <set>
#include<queue>

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

class Solution {
public:
	bool isAnagram(string s, string t) {
		sort(s.begin(), s.end());
		sort(t.begin(), t.end());
		return s == t ? true : false;
	}
};


int main() {
	
}








