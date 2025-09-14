// public class Number{
//     public static void main(String[] args){
//         byte my_byte = 124;
//         short my_short = 32564;
//         int my_int = 45784612;
//         long my_long = 46789451;
//         long result = my_byte + my_short + my_long + my_int;
//         System.out.println("结果为" + result);
//     }
// }
// public class General{
//     public static void main(String[] args){
//         char word = 'd', word_2 = '@';
//         int p = 23045, p_2 = 45213;
//         // char word = 'd';
//         // char word_2 = '@';
//         // 单引号为char(字符),双引号为String(字符串)
//         // int p = 23045;
//         // int p_2 = 45213;
//         System.out.println((int)word);
//         System.out.println((int)word_2);
//         System.out.println((char)p);
//         System.out.println((char)p_2);
//         // Java的强类型转换(目标类型)原内容     
//     }
// }

//常量与变量的声明
// public class General{
//     static final double PI = 3.14;
//     static int age = 23;
//     public static void main(String[] args){
//         final int number;
//         number = 12345;
//         age = 22;
//         System.out.println(PI);
//         System.out.println(age);
//         System.out.println(number);
//     }
// }

//当类变量和局部变量的名称相同时,在方法中生效的是局部变量
// public class General{
//     static int times = 3;
//     public static void main(String[] args) {
//         int times = 4;
//         System.out.println(times);
//     }
// }

//条件语句
// public class General{
//     public static void main(String[] args){
//         int x = 20;
//         {
//             int y = 40;
//             System.out.println(y);

//             int z = 245;

//             boolean b;
//             {
//                 b = y > z;
//                 System.out.println(b);
//             }
//         }
//         String word = "Hello JAVA";
//         System.out.println(word);
//     }
// }

// 条件选择语句
// public class General{
//     public static void main(String[] args){
//         int x = 45;
//         int y = 12;
//         if (x > y) {
//             System.out.println("X>Y");
//         }
//         else if (x < y) {
//             System.out.println("X<Y");
//         }
//         else {
//             System.err.println("条件错误！");
//         }
//         // else{
//         //     System.out.println("X<Y");
//         // }
//     }
// }

//Java的循环语句
// public class General{
//     public static void main(String[] args){
//         int x = 1;
//         int sum = 0;
//         while (x < 10) {
//             sum = sum + x;
//             x++;
//         }
//         System.out.println(sum);
//     }
// }

//Java的do-while语句:至少执行一次，先执行，后判断
// public class General{
//     public static void main(String[] args){
//         int a = 100;
//         while (a == 60) {
//             System.out.println("OK1");
//             a--;
//         }
//         int b = 100;
//         do{
//             System.out.println("OK2");
//             b--;
//         }
//         while (b == 60);
//     }
// }

//Java中的for循环:对变量的初始化必须在for循环的语句表达式中完成
// public class General{
//     public static void main(String[] args){
//         int sum = 0;
//         for (int i = 2; i <= 100; i += 2){
//             sum = sum + i;
//         }
//         System.out.println(sum);
//     }
// }
// public class General{
//     public static void main(String[] args) {
//         int arr[] = {1, 2, 3, 4};
//         for(int i = 0; i < arr.length; i++){
//             System.out.println(arr[i]);
//         }
//     }
// }

//判断奇偶
// public class General{
//     public static void main(String[] args){
//         int arr[] = {9, 8};
//         for(int i = 0; i < arr.length; i++){
//             int a = arr[i];
//             float b = a % 2;
//             if (b == 0) {
//                 System.out.println(arr[i] + "为偶数");
//             } else {
//                 System.out.println(arr[i] + "为奇数");
//             }
//         }
//     }
// }

//Java中字符串的操作
//声明字符串 String str = [null]
//创建字符串
// public class General{
//     public static void main(String[] args){
//         String s = "good";
//         System.out.println(s);
//         String s_1 = "hello";
//         System.out.println(s + " " + s_1);
//     }
// }
// public class General{
//     public static void main(String[] args){
//         String a = "good";
//         System.out.println(a.charAt(0));
//     }
// }

//Java中在任意位置截取字符串:不包括初始位置
// public class General{
//     public static void main(String[] args) {
//         String a = "good";
//         System.out.println(a.substring(3)); 
//     }
// }

//Java中检索字符串的开头与结尾
// public class General{
//     public static void main(String[] args) {
//         String num_1 = "22045612";
//         String num_2 = "21304578";
//         boolean b_1 = num_1.startsWith("22", 0);
//         boolean b_2 = num_1.endsWith("78");
//         boolean b_3 = num_2.startsWith("22", 0);
//         boolean b_4 = num_2.endsWith("78");

//         System.out.println(b_1);
//         System.out.println(b_2);
//         System.out.println(b_3);
//         System.out.println(b_4);

//     }
// }

// public class General{
//     public static void main(String[] args) {
//         String a = "bigger";
//         String b = "BIGGER";

//         System.out.println(a.toUpperCase());
//         System.out.println(b.toLowerCase());
//     }
// }

// public class General{
//     public static void main(String[] args){
//         String str_1 = "str_1";
//         String str_2 = "str_2";

//         String str_1_sub = str_1.substring(0, str_1.length()-1);
//         String str_2_sub = str_2.substring(0, str_2.length()-1);
//         System.out.println(str_1_sub + " " + str_2_sub);
//         if (str_1_sub.equals(str_2_sub)) {
//             System.err.println("两个字串相同");
//         } else {
//             System.out.println("两个字串并不相同");
//         }
//     }
// }

//Java字符串生成器
// public class General{
//     public static void main(String[] args){
//         StringBuilder new_str = new StringBuilder("str");
//         for(int i = 1; i <= 10; i++){
//             new_str.append(i);
//         }
//         new_str.toString();
//         System.out.println(new_str);
//     }
// }

// Java中的数组
//一维数组
// public class General{
//     public static void main(String[] args){
//         int day[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
//         for (int i = 0; i < day.length; i++) {
//             System.out.println((i + 1) + "月有" + day[i] + "天");
//         }
//     }
// }

// 二维数组
public class General{
    public static void main(String[] args) {
        int arr[][] = {{1, 2}, {3, 4}, {5, 6}};
        for(int i = 0; i < arr.length; i++){
            for(int a = 0; a < arr[i].length; a++){
                System.out.println(arr[i][a]);
            }
        }
    }
}