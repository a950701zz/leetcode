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
//				string sub = s.substr(i, j-i+1);  //�ַ�������j+1-i
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


//35 ��ͨ������
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
//	//���ַ�����
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


//70��¥��
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
//	int maxSubArray(vector<int>& nums) {  //maxsum ��ʾ�����±�Ϊi��Ԫ�ص��Ӵ���������
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
//			if (minprice > prices[i-1]) minprice = prices[i-1]; //ǰһ�����С�۸�
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

// ��ȫƽ����
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
//				f[i] = min(f[i], f[j*j] + f[i-j*j]);  //���û���f[i+j*j]�����f[i]+1
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


////��ת����
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


//�ƶ���
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

//�󽻼�
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
//		//��û�з��أ�������ֻ�ôӵ�i�к͵�i����ɸѡ
//		//��i��
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

//  �����
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

//ֻ����һ�ε���
//class Solution {
//public:
//	int singleNumber(vector<int>& nums) {
//		int rst = 0;
//		for (int i = 0; i < nums.size(); i++)
//			rst ^= nums[i];
//		return rst;
//	}
//};

//���Ʊ�����ʱ��2
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

//�ַ�����
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
//			}//��ǰ���ɿ��Էַ�
//			else
//				j++;
//		}
//		return count;
//	}
//};

//������С��
//class Solution {
//public:
//	string fractionToDecimal(int numerator, int denominator) {
//		long long t = numerator, d = denominator;
//		map<long  long,int > A;  //����һ��Map A��¼���ǵ�ǰ�ĳ����Լ����������ǵ�ʱans��size,Ϊ������һ�γ���
//		                        //ͬ���ĳ���ʱ�Ὺʼѭ������¼����sizeȷ���������λ�á�
//		string ans;
//		if (t*d < 0) ans = "-";
//		t = abs(t); d = abs(d);
//		ans += to_string(t / d);  //��������
//		t = t%d;  //С������ȡ����
//		if (!t) return ans;
//		ans += ".";
//		while (t)   //������Ϊ0��ʱ��ѭ������ t�ǵ�ǰ���Է�ĸ����
//		{
//			if (A.count(t))  //���ظ��ĳ���������ֹͣѭ��
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

//ȱʧ������
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

//λ1�ĸ���   һ����������ȥ1����õ��Ľ���ǰ������ұߵ�1���0
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


//�׳˺��0
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

//ʢˮ��������
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


//���ͬǰ׺
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
//		for (int i = nums.size() - 1; i >= 1;i--)  //res�������ȴ洢��ǰλ���ұ����ĳ˻�
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

//����ˮ
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
//			if (height[i] < last)  //��һ�����Ӹ߶Ƚ���
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


//Ӳ������
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
//				if (i >= coins[j])    //i�ǵ�ǰ��̬�滮��״̬������coins[j]ÿ��Ӳ��
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
//			if (num == 0)  //��һ�α���ʱ
//			{
//				if (x.count(nums[i]-1)!=0)//���������
//				{
//					if (x.count(nums[i]+1)!=0)  //���ұ���������
//					{
//						int newsequence = x[nums[i] - 1] + x[nums[i] + 1] + 1;
//						cout << (nums[i] - 1);
//						cout << x.count(nums[i] - 1);
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i] + x[nums[i] + 1]] = newsequence;
//						x[nums[i] ] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //��������ڣ��ұ߲�����
//					{
//						int newsequence = x[nums[i] - 1] + 1;
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//				}
//				else   //�����������
//				{
//					if (x.count(nums[i] + 1)!= 0) //�����������,�ұ�������
//					{
//						int newsequence = x[nums[i] + 1] + 1;
//						x[nums[i] + x[nums[i]+1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //���Ҷ�������
//						x[nums[i]] = 1;
//				}
//			}
//		}
//		return rst;
//	}
//};


//������С��������
//class Solution {
//public:
//	int minSubArrayLen(int s, vector<int>& nums) {
//		int left = 0, right = 0, sumlen = 0, rst = INT_MAX;
//		if (nums.size() == 0) return 0;
//		bool stopflag = false;
//		while (true)
//		{
//			  //��ǰ��
//			if (sumlen>s && right!=nums.size())  //����s,�󻬴����ƶ�
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
//			else  //�һ���ָ���˱߽�
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
//				int temp = matrix[i][j + roundindex];  //ÿһ����λ�������� matrix.size()-1-2*i 
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
//			} while (!(s[head] >= 'a'&&s[head] <= 'z' || s[head] >= 'A'&&s[head] <= 'Z'||s[head]>='0'&&s[head]<='9')); //head����ĸʱ
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
//		while (s.find(n)==s.end())  //set��û�е�ǰ����
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
//	int findPeakElement(vector<int>& nums) {  //���ַ��������ַ�������ָ��������ʱ����ô�����м�ڵ�����ߵ���
//		int n = nums.size();
//		if (n == 1) return 0;
//		if (n == 2) if (nums[1] > nums[0]) return 1; else return 0;
//		if (nums[n / 2] > nums[n / 2 + 1] && nums[n / 2] > nums[n / 2 - 1])
//			return n / 2;
//		if (nums[n / 2] < nums[n / 2 + 1])  //���ұ�Ѱ��
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
//		//���Ƶı���
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
//	int maxPoints(vector<Point>& points) {    //ÿһ�α���һ���㣬ͨ��һ���һ��б��ȷ��Ψһ��ֱ�ߣ�һ�α����ҵ�����ֱ��ͨ���õ�����������������е������õ������Ž⡣
//		int res = 0;
//		for (int i = 0; i < points.size(); ++i) {
//			map<pair<int, int>, int> m;
//			int duplicate = 1;
//			for (int j = i + 1; j < points.size(); ++j) {
//				if (points[i].x == points[j].x && points[i].y == points[j].y) {   //�������һ������ͬһ���㣬�����ͬһ����Ļ�Ҫ��¼��Ҳ����һ��ֱ���ϡ�
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
//		return (b == 0) ? a : gcd(b, a % b); //�����Լ�����㷨
//	}
//};

//class Solution {
//public:
//	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
//		queue<string> q;
//		map<string, int> m1; //���������е��ַ������ڲ����Ƿ����
//		map<string, int> re; //������
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
//	bool is_str;  //�жϵ�ǰ�ڵ��Ƿ�Ϊһ���������ַ���
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
//				cur = cur->next[w - 'a']; // ����curָ���ָ��ʹ��ָ����һ�����
//		}
//		return (cur != NULL&&cur->is_str); // curָ�벻Ϊ����curָ��ָ��Ľ��Ϊһ���������ַ�������ɹ��ҵ��ַ���
//	}
//
//	/** Returns if there is any word in the trie that starts with the given prefix. */
//	bool startsWith(string prefix) {
//		Trie *cur = this;
//		for (char w : prefix){
//			if (cur != NULL)
//				cur = cur->next[w - 'a'];
//		}
//		return (cur != NULL); // ���search(),����ֻ���ж�curָ���Ƿ�Ϊ�վ�����
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
//		while (!Q.empty())   //BFS��ֹѭ������������Ϊ��
//		{
//			pair<string, int>node = Q.front();
//			Q.pop();
//			string currentword = node.first;  //�ڵ��string
//			int level = node.second;   //�ڵ��level
//			for (int i = 0; i < len; i++)
//			{
//				string intermediate = currentword.substr(0, i) + "*" + currentword.substr(i + 1, len - (i + 1));
//				map<string, vector<string>>::iterator it = allComboDict.begin();
//				while (it != allComboDict.end())
//				{
//					for (int j = 0; j < (it->second).size();j++)
//					{
//						string intermediate1 = it->second[j];
//						if (intermediate == intermediate1)  //�ҵ�����ͨ���ֶ�
//						{
//							if (it->first == endWord)
//								return level + 1;
//							else
//								if (visited[it->first] == false)  //û���ʹ�
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
//	bool is_str;//��ʶ��ǰ�ڵ��Ƿ���һ���������ַ���
//	Trie *next[Maxn];  //ָ��Trie��ָ������
//	Trie() {
//		is_str = NULL;
//		memset(next, 0, sizeof(next)); //��ʼ��ָ������
//	}
//
//	/** Inserts a word into the trie. */
//	void insert(string word) {
//		Trie *cur = this; //����ָ��ǰ�ڵ�
//		for (char w : word)
//		{
//			if (cur->next[w - 'a'] == NULL)  //��ǰ�ڵ�û��ָ��ǰ�ַ�w�ķ�֧��Ҫ�½��ڵ�
//			{
//				Trie *new_node = new Trie();
//				cur->next[w - 'a'] = new_node;
//			}
//			cur = cur->next[w - 'a'];//ÿһ�ζ������һ����֧���ڵ�ָ����һ���ַ�
//		}
//		cur->is_str = true;//ָ����ĩβ���ַ��ˣ���ǰ�ڵ㴦��һ�������ĵ���
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
//	int val;  //��ǰֵ
//	int count;  //��nums�б�valС�����ĸ���
//	BSTNode *left; //������ָ��
//	BSTNode *right; // ������ָ��
//	BSTNode(int x) :val(x), left(NULL), right(NULL), count(0){};//��Ҫά��һ��count����Ϊ����ʱ����
//	//�Ὣÿ��������������Ҫ��¼ÿһ���ڵ��countֵ
//};
//
//void BST_insert(BSTNode *node, BSTNode *insert_node, int &count_small)
//{
//	//��node�ڵ㿪ʼ�����µ�insert_node�ڵ�
//	if (node->val >= insert_node->val)    //�½ڵ�С�ڵ��ھɽڵ�
//	{
//		node->count++;  //�ȵ�ǰ�ڵ�С�Ľڵ����Ҫ����
//		if (node->left)   //�½ڵ���뵽���
//			BST_insert(node->left, insert_node, count_small);
//		else
//			node->left = insert_node;
//		//����Ľڵ�ȵ�ǰ�ڵ�ҪС��
//	}
//	else
//	{
//		count_small += node->count + 1;
//		if (node->right)  //��ǰ�ڵ����������Ϊ��
//			BST_insert(node->right, insert_node, count_small);
//		else
//			//�������ǿ�
//			node->right = insert_node;
//	}
//}
//
//class Solution{
//public:
//	vector<int> countSmaller(vector<int> &nums)
//	{
//		int n = nums.size();
//		if (n == 0) return{}; //���ؿյ�ʱ�򷵻ص���һ�����飬�ô����ű�ʾ
//		vector<int>  count;  // �����������vector
//		count.push_back(0); // ���һ��Ԫ��ֵΪ0
//		BSTNode *node = new BSTNode(nums[n - 1]);  //����n-1���ڵ�,node��valΪnums[n-1]�Ľڵ�
//		int count_small;
//		for (int i = 1; i < n; i++)
//		{
//			count_small = 0; //ÿһ�ζ���nums[n-1]�Ľڵ㿪ʼ�����µĽڵ�
//			BST_insert(node, new BSTNode(nums[n - i - 1]), count_small);//����nums[n-i-1]��Ԫ�ص���������
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
//	//����ָ��Ļ���˼���������ָ��֮��ʼ��Ϊ��ָ�����ߵ�·�̣���ָ�������Ժ���ָ��һ�������������Ļ�����
//	// ��Ϊ��ʼ����m����m�Ǵ���㵽���ľ��룬���Դ�ʱ��ָ������m���͵��˻������
//	// ��ʱ�ÿ�ָ���ͷ�ߣ���m����Ҳ������㣬��ָ�������������Ҳ�����ظ�������
//	int findDuplicate(vector<int>& nums) {
//		int fast = nums[0];
//		int slow = nums[0];  //��nums[0]������ʼλ�ã���ָ������ʱҪ�ص�nums[0]
//		do 
//		{
//			slow = nums[slow];  //��һ��
//			fast = nums[nums[fast]]; //��ǰ������
//		} while (fast!=slow);
//
//		// �������ָ��ص����
//		fast = nums[0];
//		while (fast != slow)  //����Ѿ���������ôֱ�ӷ��أ�������do while
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
//		//x�ǵ�һ��vector y�ǵڶ���vector��index
//		if (xx < 0 || xx >= x || yy < 0 || yy >= y)  //����Խ�磬����
//			return;
//		else
//			if (visited[xx][yy] == true || grid[xx][yy] == '0')  //����������
//				return;
//		  else
//		{
//			grid[xx][yy] = '0';
//			visited[xx][yy] = true;//����Ϊ���ʹ�
//			BFS(grid, xx + 1, yy, visited);
//			BFS(grid, xx - 1, yy, visited);
//			BFS(grid, xx , yy + 1, visited);
//			BFS(grid, xx , yy - 1, visited);
//		}
//	}
//};
//class Solution {
//public:  //�����зֿ����������Խ��ײ��ˡ���Ҫ���������ע��ŷ� �����ظ�����
//	void wiggleSort(vector<int>& nums) {
//		vector<int> copyed(nums);
//		if (nums.size() == 0) return ;
//		sort(copyed.begin(), copyed.end());
//		int len = nums.size()-1;
//		for (int i = 1; i < nums.size();i=i+2)
//	           //�����Ŵ��� 
//				nums[i] = copyed[len--];
//		
//		for (int i = 0; i < nums.size(); i=i+2)
//			 //ż����С�� 
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


//// ����������������������ܿ��ѧ���ˣ�����˴򿨣�����ǿ�磬���ǲ��������̲� haha~
// struct TreeNode {
//     int val;
//     TreeNode *left;
//     TreeNode *right;
//     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// };
// //�����ض��������������ʵ���Ͼ��Ǳ����õ������δ�С�������еĽڵ�ֵ
//class Solution {
//public:
//	vector<int>result;
//	int  kthSmallest(TreeNode* root,int k) {
//		if (root == NULL) return 0;
//		if (root->left != NULL)  //����з������
//			kthSmallest(root->left,1);
//		//���û�У������Լ��ˣ����Է�������ֵ������ұ�
//		result.push_back(root->val);
//		if (root->right != NULL) kthSmallest(root->right,1);
//		return result[k-1];
//	}
//};




////��Ҫ������ݹ��˼·�����뵽��Ҫ���մ��˼·��Ҫ����ϸ֦ĩ�ڣ�Ҫ�����Ʋ������ˡ��öิϰ���Ρ�
//class Solution {
//public:
//	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
//		if (root == p || root == q) return root;//�ҵ���ָ���ڵ�ֱ�ӷ���
//		if (root == NULL) return NULL; //��Ѱ�����û����Ѱ����˵��û�ҵ�������NULL
//		//û���ҵ���Ҫ��ʼ��Ѱ��,�������
//		TreeNode *left = lowestCommonAncestor(root->left, p, q);
//		TreeNode *right = lowestCommonAncestor(root->right, p, q);
//
//		//����ԭ��
//		if (left == NULL&&right ==NULL) //��û������ 
//			return NULL;
//		if (left != NULL&&right == NULL)  //������û��
//			return left;
//		if (left == NULL&&right != NULL)
//			return right;
//		else return root; //���ж���
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
//			if (root->val > p->val&&root->val > q->val) //��С�ڵ�ǰ�ڵ㣬�϶����ұ�
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


//��̬�滮�������������ΪС�����⡣dp[i]��ʾ������iλΪ�����е�����������������
//������һ״̬���� dp[i+1] Ҫ���£�nums[i+1]��Ҫ���κ�nums[i],nums[i-1]...nums[0]�Ƚ�������dp[i+1]
//����̬�滮��ʱ������Ҫ���߼����������д���롣
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
//				//����i-2��0,�ҵ�dp[i]
//				if (nums[i] < nums[j]) median = 1;
//				else 
//					if (nums[i]>nums[j]) 
//				{
//					median = dp[j] + 1;
//				}
//				else
//				{   //�������
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
		result = INT_MIN;   // ����ȫ���ڵ�Ϊ���������
		getPath(root);
		return result;
	}
	int getPath(TreeNode* node) {
		if (node == NULL) {
			return 0;
		}
		int left = getPath(node->left);
		int right = getPath(node->right);
		int tmp = max(max(left + node->val, right + node->val), node->val); // ����������Ǿ����ڵ�node�ҿ���������ϵģ���Ҫ���ظ��ϲ�ʹ��
		result = max(result, max(tmp, left + right + node->val));    // ����������ϵ����ֻ��Ҫ���ڸ��½�����������Ϸ���
		return tmp;
	}
};

int main() {

}








