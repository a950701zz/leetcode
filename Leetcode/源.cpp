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

struct TreeNode {
	*int val;
	*TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
	int result;
	int maxPathSum(TreeNode* root) {
		result = INT_MIN;   // 考虑全部节点为负数的情况
		getPath(root);
		return result;
	}
	int getPath(TreeNode* node) {
		if (node == NULL) {
			return 0;
		}
		int left = getPath(node->left);
		int right = getPath(node->right);
		int tmp = max(max(left + node->val, right + node->val), node->val); // 这三种情况是经过节点node且可以向上组合的，需要返回给上层使用
		result = max(result, max(tmp, left + right + node->val));    // 不能向上组合的情况只需要用于更新结果，无需向上返回
		return tmp;
	}
};

int main() {

}








