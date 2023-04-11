begin_version
3
end_version
begin_metric
0
end_metric
18
begin_variable
var0
-1
2
Atom power_on(instrument6)
NegatedAtom power_on(instrument6)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument7)
NegatedAtom power_on(instrument7)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite3)
NegatedAtom power_avail(satellite3)
end_variable
begin_variable
var3
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var5
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var6
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var7
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var8
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var9
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var10
-1
16
Atom pointing(satellite3, groundstation1)
Atom pointing(satellite3, groundstation10)
Atom pointing(satellite3, groundstation13)
Atom pointing(satellite3, groundstation14)
Atom pointing(satellite3, groundstation2)
Atom pointing(satellite3, groundstation3)
Atom pointing(satellite3, groundstation5)
Atom pointing(satellite3, groundstation6)
Atom pointing(satellite3, groundstation8)
Atom pointing(satellite3, planet15)
Atom pointing(satellite3, star0)
Atom pointing(satellite3, star11)
Atom pointing(satellite3, star12)
Atom pointing(satellite3, star4)
Atom pointing(satellite3, star7)
Atom pointing(satellite3, star9)
end_variable
begin_variable
var11
-1
16
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation10)
Atom pointing(satellite1, groundstation13)
Atom pointing(satellite1, groundstation14)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation5)
Atom pointing(satellite1, groundstation6)
Atom pointing(satellite1, groundstation8)
Atom pointing(satellite1, planet15)
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star11)
Atom pointing(satellite1, star12)
Atom pointing(satellite1, star4)
Atom pointing(satellite1, star7)
Atom pointing(satellite1, star9)
end_variable
begin_variable
var12
-1
16
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation10)
Atom pointing(satellite0, groundstation13)
Atom pointing(satellite0, groundstation14)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation5)
Atom pointing(satellite0, groundstation6)
Atom pointing(satellite0, groundstation8)
Atom pointing(satellite0, planet15)
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star11)
Atom pointing(satellite0, star12)
Atom pointing(satellite0, star4)
Atom pointing(satellite0, star7)
Atom pointing(satellite0, star9)
end_variable
begin_variable
var13
-1
2
Atom calibrated(instrument7)
NegatedAtom calibrated(instrument7)
end_variable
begin_variable
var14
-1
2
Atom calibrated(instrument6)
NegatedAtom calibrated(instrument6)
end_variable
begin_variable
var15
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var16
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var17
-1
2
Atom have_image(planet15, spectrograph1)
NegatedAtom have_image(planet15, spectrograph1)
end_variable
0
begin_state
1
1
0
1
1
1
0
1
1
0
6
7
7
1
1
1
1
1
end_state
begin_goal
1
17 0
end_goal
747
begin_operator
calibrate satellite0 instrument1 star9
2
12 15
8 0
1
0 16 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 groundstation3
2
11 5
4 0
1
0 15 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star0
2
11 10
4 0
1
0 15 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star9
2
11 15
4 0
1
0 15 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument6 groundstation14
2
10 3
0 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument6 groundstation6
2
10 7
0 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument6 star0
2
10 10
0 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument7 groundstation14
2
10 3
1 0
1
0 13 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument7 groundstation5
2
10 6
1 0
1
0 13 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 9 -1 0
0 7 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 9 -1 0
0 8 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 6 -1 0
0 3 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 6 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite1
0
2
0 6 -1 0
0 5 0 1
1
end_operator
begin_operator
switch_off instrument6 satellite3
0
2
0 2 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument7 satellite3
0
2
0 2 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
2
0 9 0 1
0 7 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 16 -1 1
0 9 0 1
0 8 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
2
0 6 0 1
0 3 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 15 -1 1
0 6 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite1
0
2
0 6 0 1
0 5 -1 0
1
end_operator
begin_operator
switch_on instrument6 satellite3
0
3
0 14 -1 1
0 2 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument7 satellite3
0
3
0 13 -1 1
0 2 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 planet15 instrument1 spectrograph1
3
16 0
12 9
8 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 planet15 instrument3 spectrograph1
3
15 0
11 9
4 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite3 planet15 instrument6 spectrograph1
3
14 0
10 9
0 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite3 planet15 instrument7 spectrograph1
3
13 0
10 9
1 0
1
0 17 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation10
0
1
0 12 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation13
0
1
0 12 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation14
0
1
0 12 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation2
0
1
0 12 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation3
0
1
0 12 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation5
0
1
0 12 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation6
0
1
0 12 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation8
0
1
0 12 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 planet15
0
1
0 12 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star0
0
1
0 12 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star11
0
1
0 12 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star12
0
1
0 12 12 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star4
0
1
0 12 13 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star7
0
1
0 12 14 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star9
0
1
0 12 15 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation1
0
1
0 12 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation13
0
1
0 12 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation14
0
1
0 12 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation2
0
1
0 12 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation3
0
1
0 12 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation5
0
1
0 12 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation6
0
1
0 12 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation8
0
1
0 12 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 planet15
0
1
0 12 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star0
0
1
0 12 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star11
0
1
0 12 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star12
0
1
0 12 12 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star4
0
1
0 12 13 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star7
0
1
0 12 14 1
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star9
0
1
0 12 15 1
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation1
0
1
0 12 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation10
0
1
0 12 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation14
0
1
0 12 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation2
0
1
0 12 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation3
0
1
0 12 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation5
0
1
0 12 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation6
0
1
0 12 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 groundstation8
0
1
0 12 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 planet15
0
1
0 12 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star0
0
1
0 12 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star11
0
1
0 12 11 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star12
0
1
0 12 12 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star4
0
1
0 12 13 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star7
0
1
0 12 14 2
1
end_operator
begin_operator
turn_to satellite0 groundstation13 star9
0
1
0 12 15 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation1
0
1
0 12 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation10
0
1
0 12 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation13
0
1
0 12 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation2
0
1
0 12 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation3
0
1
0 12 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation5
0
1
0 12 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation6
0
1
0 12 7 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation8
0
1
0 12 8 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 planet15
0
1
0 12 9 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star0
0
1
0 12 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star11
0
1
0 12 11 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star12
0
1
0 12 12 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star4
0
1
0 12 13 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star7
0
1
0 12 14 3
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star9
0
1
0 12 15 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation1
0
1
0 12 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation10
0
1
0 12 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation13
0
1
0 12 2 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation14
0
1
0 12 3 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation3
0
1
0 12 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation5
0
1
0 12 6 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation6
0
1
0 12 7 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation8
0
1
0 12 8 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 planet15
0
1
0 12 9 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star0
0
1
0 12 10 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star11
0
1
0 12 11 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star12
0
1
0 12 12 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star4
0
1
0 12 13 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star7
0
1
0 12 14 4
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star9
0
1
0 12 15 4
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation1
0
1
0 12 0 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation10
0
1
0 12 1 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation13
0
1
0 12 2 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation14
0
1
0 12 3 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation2
0
1
0 12 4 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation5
0
1
0 12 6 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation6
0
1
0 12 7 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation8
0
1
0 12 8 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet15
0
1
0 12 9 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star0
0
1
0 12 10 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star11
0
1
0 12 11 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star12
0
1
0 12 12 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star4
0
1
0 12 13 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star7
0
1
0 12 14 5
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star9
0
1
0 12 15 5
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation1
0
1
0 12 0 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation10
0
1
0 12 1 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation13
0
1
0 12 2 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation14
0
1
0 12 3 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation2
0
1
0 12 4 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation3
0
1
0 12 5 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation6
0
1
0 12 7 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation8
0
1
0 12 8 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 planet15
0
1
0 12 9 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star0
0
1
0 12 10 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star11
0
1
0 12 11 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star12
0
1
0 12 12 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star4
0
1
0 12 13 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star7
0
1
0 12 14 6
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star9
0
1
0 12 15 6
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation1
0
1
0 12 0 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation10
0
1
0 12 1 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation13
0
1
0 12 2 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation14
0
1
0 12 3 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation2
0
1
0 12 4 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation3
0
1
0 12 5 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation5
0
1
0 12 6 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation8
0
1
0 12 8 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 planet15
0
1
0 12 9 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star0
0
1
0 12 10 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star11
0
1
0 12 11 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star12
0
1
0 12 12 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star4
0
1
0 12 13 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star7
0
1
0 12 14 7
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star9
0
1
0 12 15 7
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation1
0
1
0 12 0 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation10
0
1
0 12 1 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation13
0
1
0 12 2 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation14
0
1
0 12 3 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation2
0
1
0 12 4 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation3
0
1
0 12 5 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation5
0
1
0 12 6 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation6
0
1
0 12 7 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 planet15
0
1
0 12 9 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star0
0
1
0 12 10 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star11
0
1
0 12 11 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star12
0
1
0 12 12 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star4
0
1
0 12 13 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star7
0
1
0 12 14 8
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star9
0
1
0 12 15 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation1
0
1
0 12 0 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation10
0
1
0 12 1 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation13
0
1
0 12 2 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation14
0
1
0 12 3 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation2
0
1
0 12 4 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation3
0
1
0 12 5 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation5
0
1
0 12 6 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation6
0
1
0 12 7 9
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation8
0
1
0 12 8 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star0
0
1
0 12 10 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star11
0
1
0 12 11 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star12
0
1
0 12 12 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star4
0
1
0 12 13 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star7
0
1
0 12 14 9
1
end_operator
begin_operator
turn_to satellite0 planet15 star9
0
1
0 12 15 9
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation1
0
1
0 12 0 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation10
0
1
0 12 1 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation13
0
1
0 12 2 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation14
0
1
0 12 3 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation2
0
1
0 12 4 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation3
0
1
0 12 5 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation5
0
1
0 12 6 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation6
0
1
0 12 7 10
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation8
0
1
0 12 8 10
1
end_operator
begin_operator
turn_to satellite0 star0 planet15
0
1
0 12 9 10
1
end_operator
begin_operator
turn_to satellite0 star0 star11
0
1
0 12 11 10
1
end_operator
begin_operator
turn_to satellite0 star0 star12
0
1
0 12 12 10
1
end_operator
begin_operator
turn_to satellite0 star0 star4
0
1
0 12 13 10
1
end_operator
begin_operator
turn_to satellite0 star0 star7
0
1
0 12 14 10
1
end_operator
begin_operator
turn_to satellite0 star0 star9
0
1
0 12 15 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation1
0
1
0 12 0 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation10
0
1
0 12 1 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation13
0
1
0 12 2 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation14
0
1
0 12 3 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation2
0
1
0 12 4 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation3
0
1
0 12 5 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation5
0
1
0 12 6 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation6
0
1
0 12 7 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation8
0
1
0 12 8 11
1
end_operator
begin_operator
turn_to satellite0 star11 planet15
0
1
0 12 9 11
1
end_operator
begin_operator
turn_to satellite0 star11 star0
0
1
0 12 10 11
1
end_operator
begin_operator
turn_to satellite0 star11 star12
0
1
0 12 12 11
1
end_operator
begin_operator
turn_to satellite0 star11 star4
0
1
0 12 13 11
1
end_operator
begin_operator
turn_to satellite0 star11 star7
0
1
0 12 14 11
1
end_operator
begin_operator
turn_to satellite0 star11 star9
0
1
0 12 15 11
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation1
0
1
0 12 0 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation10
0
1
0 12 1 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation13
0
1
0 12 2 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation14
0
1
0 12 3 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation2
0
1
0 12 4 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation3
0
1
0 12 5 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation5
0
1
0 12 6 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation6
0
1
0 12 7 12
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation8
0
1
0 12 8 12
1
end_operator
begin_operator
turn_to satellite0 star12 planet15
0
1
0 12 9 12
1
end_operator
begin_operator
turn_to satellite0 star12 star0
0
1
0 12 10 12
1
end_operator
begin_operator
turn_to satellite0 star12 star11
0
1
0 12 11 12
1
end_operator
begin_operator
turn_to satellite0 star12 star4
0
1
0 12 13 12
1
end_operator
begin_operator
turn_to satellite0 star12 star7
0
1
0 12 14 12
1
end_operator
begin_operator
turn_to satellite0 star12 star9
0
1
0 12 15 12
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation1
0
1
0 12 0 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation10
0
1
0 12 1 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation13
0
1
0 12 2 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation14
0
1
0 12 3 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation2
0
1
0 12 4 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation3
0
1
0 12 5 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation5
0
1
0 12 6 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation6
0
1
0 12 7 13
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation8
0
1
0 12 8 13
1
end_operator
begin_operator
turn_to satellite0 star4 planet15
0
1
0 12 9 13
1
end_operator
begin_operator
turn_to satellite0 star4 star0
0
1
0 12 10 13
1
end_operator
begin_operator
turn_to satellite0 star4 star11
0
1
0 12 11 13
1
end_operator
begin_operator
turn_to satellite0 star4 star12
0
1
0 12 12 13
1
end_operator
begin_operator
turn_to satellite0 star4 star7
0
1
0 12 14 13
1
end_operator
begin_operator
turn_to satellite0 star4 star9
0
1
0 12 15 13
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation1
0
1
0 12 0 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation10
0
1
0 12 1 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation13
0
1
0 12 2 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation14
0
1
0 12 3 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation2
0
1
0 12 4 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation3
0
1
0 12 5 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation5
0
1
0 12 6 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation6
0
1
0 12 7 14
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation8
0
1
0 12 8 14
1
end_operator
begin_operator
turn_to satellite0 star7 planet15
0
1
0 12 9 14
1
end_operator
begin_operator
turn_to satellite0 star7 star0
0
1
0 12 10 14
1
end_operator
begin_operator
turn_to satellite0 star7 star11
0
1
0 12 11 14
1
end_operator
begin_operator
turn_to satellite0 star7 star12
0
1
0 12 12 14
1
end_operator
begin_operator
turn_to satellite0 star7 star4
0
1
0 12 13 14
1
end_operator
begin_operator
turn_to satellite0 star7 star9
0
1
0 12 15 14
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation1
0
1
0 12 0 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation10
0
1
0 12 1 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation13
0
1
0 12 2 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation14
0
1
0 12 3 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation2
0
1
0 12 4 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation3
0
1
0 12 5 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation5
0
1
0 12 6 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation6
0
1
0 12 7 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation8
0
1
0 12 8 15
1
end_operator
begin_operator
turn_to satellite0 star9 planet15
0
1
0 12 9 15
1
end_operator
begin_operator
turn_to satellite0 star9 star0
0
1
0 12 10 15
1
end_operator
begin_operator
turn_to satellite0 star9 star11
0
1
0 12 11 15
1
end_operator
begin_operator
turn_to satellite0 star9 star12
0
1
0 12 12 15
1
end_operator
begin_operator
turn_to satellite0 star9 star4
0
1
0 12 13 15
1
end_operator
begin_operator
turn_to satellite0 star9 star7
0
1
0 12 14 15
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation10
0
1
0 11 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation13
0
1
0 11 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation14
0
1
0 11 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation2
0
1
0 11 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation3
0
1
0 11 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation5
0
1
0 11 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation6
0
1
0 11 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation8
0
1
0 11 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 planet15
0
1
0 11 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star0
0
1
0 11 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star11
0
1
0 11 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star12
0
1
0 11 12 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star4
0
1
0 11 13 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star7
0
1
0 11 14 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star9
0
1
0 11 15 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation1
0
1
0 11 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation13
0
1
0 11 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation14
0
1
0 11 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation2
0
1
0 11 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation3
0
1
0 11 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation5
0
1
0 11 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation6
0
1
0 11 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation8
0
1
0 11 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 planet15
0
1
0 11 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star0
0
1
0 11 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star11
0
1
0 11 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star12
0
1
0 11 12 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star4
0
1
0 11 13 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star7
0
1
0 11 14 1
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star9
0
1
0 11 15 1
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation1
0
1
0 11 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation10
0
1
0 11 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation14
0
1
0 11 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation2
0
1
0 11 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation3
0
1
0 11 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation5
0
1
0 11 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation6
0
1
0 11 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 groundstation8
0
1
0 11 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 planet15
0
1
0 11 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star0
0
1
0 11 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star11
0
1
0 11 11 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star12
0
1
0 11 12 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star4
0
1
0 11 13 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star7
0
1
0 11 14 2
1
end_operator
begin_operator
turn_to satellite1 groundstation13 star9
0
1
0 11 15 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation1
0
1
0 11 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation10
0
1
0 11 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation13
0
1
0 11 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation2
0
1
0 11 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation3
0
1
0 11 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation5
0
1
0 11 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation6
0
1
0 11 7 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation8
0
1
0 11 8 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 planet15
0
1
0 11 9 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star0
0
1
0 11 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star11
0
1
0 11 11 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star12
0
1
0 11 12 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star4
0
1
0 11 13 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star7
0
1
0 11 14 3
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star9
0
1
0 11 15 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation1
0
1
0 11 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation10
0
1
0 11 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation13
0
1
0 11 2 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation14
0
1
0 11 3 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation3
0
1
0 11 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation5
0
1
0 11 6 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation6
0
1
0 11 7 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation8
0
1
0 11 8 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 planet15
0
1
0 11 9 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star0
0
1
0 11 10 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star11
0
1
0 11 11 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star12
0
1
0 11 12 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star4
0
1
0 11 13 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star7
0
1
0 11 14 4
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star9
0
1
0 11 15 4
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation1
0
1
0 11 0 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation10
0
1
0 11 1 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation13
0
1
0 11 2 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation14
0
1
0 11 3 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation2
0
1
0 11 4 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation5
0
1
0 11 6 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation6
0
1
0 11 7 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation8
0
1
0 11 8 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet15
0
1
0 11 9 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star0
0
1
0 11 10 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star11
0
1
0 11 11 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star12
0
1
0 11 12 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star4
0
1
0 11 13 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star7
0
1
0 11 14 5
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star9
0
1
0 11 15 5
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation1
0
1
0 11 0 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation10
0
1
0 11 1 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation13
0
1
0 11 2 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation14
0
1
0 11 3 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation2
0
1
0 11 4 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation3
0
1
0 11 5 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation6
0
1
0 11 7 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation8
0
1
0 11 8 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 planet15
0
1
0 11 9 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star0
0
1
0 11 10 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star11
0
1
0 11 11 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star12
0
1
0 11 12 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star4
0
1
0 11 13 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star7
0
1
0 11 14 6
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star9
0
1
0 11 15 6
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation1
0
1
0 11 0 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation10
0
1
0 11 1 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation13
0
1
0 11 2 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation14
0
1
0 11 3 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation2
0
1
0 11 4 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation3
0
1
0 11 5 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation5
0
1
0 11 6 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation8
0
1
0 11 8 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 planet15
0
1
0 11 9 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star0
0
1
0 11 10 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star11
0
1
0 11 11 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star12
0
1
0 11 12 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star4
0
1
0 11 13 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star7
0
1
0 11 14 7
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star9
0
1
0 11 15 7
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation1
0
1
0 11 0 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation10
0
1
0 11 1 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation13
0
1
0 11 2 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation14
0
1
0 11 3 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation2
0
1
0 11 4 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation3
0
1
0 11 5 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation5
0
1
0 11 6 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation6
0
1
0 11 7 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 planet15
0
1
0 11 9 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star0
0
1
0 11 10 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star11
0
1
0 11 11 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star12
0
1
0 11 12 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star4
0
1
0 11 13 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star7
0
1
0 11 14 8
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star9
0
1
0 11 15 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation1
0
1
0 11 0 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation10
0
1
0 11 1 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation13
0
1
0 11 2 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation14
0
1
0 11 3 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation2
0
1
0 11 4 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation3
0
1
0 11 5 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation5
0
1
0 11 6 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation6
0
1
0 11 7 9
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation8
0
1
0 11 8 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star0
0
1
0 11 10 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star11
0
1
0 11 11 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star12
0
1
0 11 12 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star4
0
1
0 11 13 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star7
0
1
0 11 14 9
1
end_operator
begin_operator
turn_to satellite1 planet15 star9
0
1
0 11 15 9
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation1
0
1
0 11 0 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation10
0
1
0 11 1 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation13
0
1
0 11 2 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation14
0
1
0 11 3 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation2
0
1
0 11 4 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation3
0
1
0 11 5 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation5
0
1
0 11 6 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation6
0
1
0 11 7 10
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation8
0
1
0 11 8 10
1
end_operator
begin_operator
turn_to satellite1 star0 planet15
0
1
0 11 9 10
1
end_operator
begin_operator
turn_to satellite1 star0 star11
0
1
0 11 11 10
1
end_operator
begin_operator
turn_to satellite1 star0 star12
0
1
0 11 12 10
1
end_operator
begin_operator
turn_to satellite1 star0 star4
0
1
0 11 13 10
1
end_operator
begin_operator
turn_to satellite1 star0 star7
0
1
0 11 14 10
1
end_operator
begin_operator
turn_to satellite1 star0 star9
0
1
0 11 15 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation1
0
1
0 11 0 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation10
0
1
0 11 1 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation13
0
1
0 11 2 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation14
0
1
0 11 3 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation2
0
1
0 11 4 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation3
0
1
0 11 5 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation5
0
1
0 11 6 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation6
0
1
0 11 7 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation8
0
1
0 11 8 11
1
end_operator
begin_operator
turn_to satellite1 star11 planet15
0
1
0 11 9 11
1
end_operator
begin_operator
turn_to satellite1 star11 star0
0
1
0 11 10 11
1
end_operator
begin_operator
turn_to satellite1 star11 star12
0
1
0 11 12 11
1
end_operator
begin_operator
turn_to satellite1 star11 star4
0
1
0 11 13 11
1
end_operator
begin_operator
turn_to satellite1 star11 star7
0
1
0 11 14 11
1
end_operator
begin_operator
turn_to satellite1 star11 star9
0
1
0 11 15 11
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation1
0
1
0 11 0 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation10
0
1
0 11 1 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation13
0
1
0 11 2 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation14
0
1
0 11 3 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation2
0
1
0 11 4 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation3
0
1
0 11 5 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation5
0
1
0 11 6 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation6
0
1
0 11 7 12
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation8
0
1
0 11 8 12
1
end_operator
begin_operator
turn_to satellite1 star12 planet15
0
1
0 11 9 12
1
end_operator
begin_operator
turn_to satellite1 star12 star0
0
1
0 11 10 12
1
end_operator
begin_operator
turn_to satellite1 star12 star11
0
1
0 11 11 12
1
end_operator
begin_operator
turn_to satellite1 star12 star4
0
1
0 11 13 12
1
end_operator
begin_operator
turn_to satellite1 star12 star7
0
1
0 11 14 12
1
end_operator
begin_operator
turn_to satellite1 star12 star9
0
1
0 11 15 12
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation1
0
1
0 11 0 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation10
0
1
0 11 1 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation13
0
1
0 11 2 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation14
0
1
0 11 3 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation2
0
1
0 11 4 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation3
0
1
0 11 5 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation5
0
1
0 11 6 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation6
0
1
0 11 7 13
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation8
0
1
0 11 8 13
1
end_operator
begin_operator
turn_to satellite1 star4 planet15
0
1
0 11 9 13
1
end_operator
begin_operator
turn_to satellite1 star4 star0
0
1
0 11 10 13
1
end_operator
begin_operator
turn_to satellite1 star4 star11
0
1
0 11 11 13
1
end_operator
begin_operator
turn_to satellite1 star4 star12
0
1
0 11 12 13
1
end_operator
begin_operator
turn_to satellite1 star4 star7
0
1
0 11 14 13
1
end_operator
begin_operator
turn_to satellite1 star4 star9
0
1
0 11 15 13
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation1
0
1
0 11 0 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation10
0
1
0 11 1 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation13
0
1
0 11 2 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation14
0
1
0 11 3 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation2
0
1
0 11 4 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation3
0
1
0 11 5 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation5
0
1
0 11 6 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation6
0
1
0 11 7 14
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation8
0
1
0 11 8 14
1
end_operator
begin_operator
turn_to satellite1 star7 planet15
0
1
0 11 9 14
1
end_operator
begin_operator
turn_to satellite1 star7 star0
0
1
0 11 10 14
1
end_operator
begin_operator
turn_to satellite1 star7 star11
0
1
0 11 11 14
1
end_operator
begin_operator
turn_to satellite1 star7 star12
0
1
0 11 12 14
1
end_operator
begin_operator
turn_to satellite1 star7 star4
0
1
0 11 13 14
1
end_operator
begin_operator
turn_to satellite1 star7 star9
0
1
0 11 15 14
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation1
0
1
0 11 0 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation10
0
1
0 11 1 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation13
0
1
0 11 2 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation14
0
1
0 11 3 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation2
0
1
0 11 4 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation3
0
1
0 11 5 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation5
0
1
0 11 6 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation6
0
1
0 11 7 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation8
0
1
0 11 8 15
1
end_operator
begin_operator
turn_to satellite1 star9 planet15
0
1
0 11 9 15
1
end_operator
begin_operator
turn_to satellite1 star9 star0
0
1
0 11 10 15
1
end_operator
begin_operator
turn_to satellite1 star9 star11
0
1
0 11 11 15
1
end_operator
begin_operator
turn_to satellite1 star9 star12
0
1
0 11 12 15
1
end_operator
begin_operator
turn_to satellite1 star9 star4
0
1
0 11 13 15
1
end_operator
begin_operator
turn_to satellite1 star9 star7
0
1
0 11 14 15
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation10
0
1
0 10 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation13
0
1
0 10 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation14
0
1
0 10 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation2
0
1
0 10 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation3
0
1
0 10 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation5
0
1
0 10 6 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation6
0
1
0 10 7 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation8
0
1
0 10 8 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 planet15
0
1
0 10 9 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star0
0
1
0 10 10 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star11
0
1
0 10 11 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star12
0
1
0 10 12 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star4
0
1
0 10 13 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star7
0
1
0 10 14 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star9
0
1
0 10 15 0
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation1
0
1
0 10 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation13
0
1
0 10 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation14
0
1
0 10 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation2
0
1
0 10 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation3
0
1
0 10 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation5
0
1
0 10 6 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation6
0
1
0 10 7 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 groundstation8
0
1
0 10 8 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 planet15
0
1
0 10 9 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star0
0
1
0 10 10 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star11
0
1
0 10 11 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star12
0
1
0 10 12 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star4
0
1
0 10 13 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star7
0
1
0 10 14 1
1
end_operator
begin_operator
turn_to satellite3 groundstation10 star9
0
1
0 10 15 1
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation1
0
1
0 10 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation10
0
1
0 10 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation14
0
1
0 10 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation2
0
1
0 10 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation3
0
1
0 10 5 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation5
0
1
0 10 6 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation6
0
1
0 10 7 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 groundstation8
0
1
0 10 8 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 planet15
0
1
0 10 9 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star0
0
1
0 10 10 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star11
0
1
0 10 11 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star12
0
1
0 10 12 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star4
0
1
0 10 13 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star7
0
1
0 10 14 2
1
end_operator
begin_operator
turn_to satellite3 groundstation13 star9
0
1
0 10 15 2
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation1
0
1
0 10 0 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation10
0
1
0 10 1 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation13
0
1
0 10 2 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation2
0
1
0 10 4 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation3
0
1
0 10 5 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation5
0
1
0 10 6 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation6
0
1
0 10 7 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 groundstation8
0
1
0 10 8 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 planet15
0
1
0 10 9 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star0
0
1
0 10 10 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star11
0
1
0 10 11 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star12
0
1
0 10 12 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star4
0
1
0 10 13 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star7
0
1
0 10 14 3
1
end_operator
begin_operator
turn_to satellite3 groundstation14 star9
0
1
0 10 15 3
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation1
0
1
0 10 0 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation10
0
1
0 10 1 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation13
0
1
0 10 2 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation14
0
1
0 10 3 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation3
0
1
0 10 5 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation5
0
1
0 10 6 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation6
0
1
0 10 7 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation8
0
1
0 10 8 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 planet15
0
1
0 10 9 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star0
0
1
0 10 10 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star11
0
1
0 10 11 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star12
0
1
0 10 12 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star4
0
1
0 10 13 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star7
0
1
0 10 14 4
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star9
0
1
0 10 15 4
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation1
0
1
0 10 0 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation10
0
1
0 10 1 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation13
0
1
0 10 2 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation14
0
1
0 10 3 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation2
0
1
0 10 4 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation5
0
1
0 10 6 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation6
0
1
0 10 7 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation8
0
1
0 10 8 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 planet15
0
1
0 10 9 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star0
0
1
0 10 10 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star11
0
1
0 10 11 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star12
0
1
0 10 12 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star4
0
1
0 10 13 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star7
0
1
0 10 14 5
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star9
0
1
0 10 15 5
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation1
0
1
0 10 0 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation10
0
1
0 10 1 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation13
0
1
0 10 2 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation14
0
1
0 10 3 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation2
0
1
0 10 4 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation3
0
1
0 10 5 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation6
0
1
0 10 7 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 groundstation8
0
1
0 10 8 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 planet15
0
1
0 10 9 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star0
0
1
0 10 10 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star11
0
1
0 10 11 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star12
0
1
0 10 12 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star4
0
1
0 10 13 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star7
0
1
0 10 14 6
1
end_operator
begin_operator
turn_to satellite3 groundstation5 star9
0
1
0 10 15 6
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation1
0
1
0 10 0 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation10
0
1
0 10 1 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation13
0
1
0 10 2 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation14
0
1
0 10 3 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation2
0
1
0 10 4 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation3
0
1
0 10 5 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation5
0
1
0 10 6 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation8
0
1
0 10 8 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 planet15
0
1
0 10 9 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star0
0
1
0 10 10 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star11
0
1
0 10 11 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star12
0
1
0 10 12 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star4
0
1
0 10 13 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star7
0
1
0 10 14 7
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star9
0
1
0 10 15 7
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation1
0
1
0 10 0 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation10
0
1
0 10 1 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation13
0
1
0 10 2 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation14
0
1
0 10 3 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation2
0
1
0 10 4 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation3
0
1
0 10 5 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation5
0
1
0 10 6 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 groundstation6
0
1
0 10 7 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 planet15
0
1
0 10 9 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star0
0
1
0 10 10 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star11
0
1
0 10 11 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star12
0
1
0 10 12 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star4
0
1
0 10 13 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star7
0
1
0 10 14 8
1
end_operator
begin_operator
turn_to satellite3 groundstation8 star9
0
1
0 10 15 8
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation1
0
1
0 10 0 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation10
0
1
0 10 1 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation13
0
1
0 10 2 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation14
0
1
0 10 3 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation2
0
1
0 10 4 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation3
0
1
0 10 5 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation5
0
1
0 10 6 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation6
0
1
0 10 7 9
1
end_operator
begin_operator
turn_to satellite3 planet15 groundstation8
0
1
0 10 8 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star0
0
1
0 10 10 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star11
0
1
0 10 11 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star12
0
1
0 10 12 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star4
0
1
0 10 13 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star7
0
1
0 10 14 9
1
end_operator
begin_operator
turn_to satellite3 planet15 star9
0
1
0 10 15 9
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation1
0
1
0 10 0 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation10
0
1
0 10 1 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation13
0
1
0 10 2 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation14
0
1
0 10 3 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation2
0
1
0 10 4 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation3
0
1
0 10 5 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation5
0
1
0 10 6 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation6
0
1
0 10 7 10
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation8
0
1
0 10 8 10
1
end_operator
begin_operator
turn_to satellite3 star0 planet15
0
1
0 10 9 10
1
end_operator
begin_operator
turn_to satellite3 star0 star11
0
1
0 10 11 10
1
end_operator
begin_operator
turn_to satellite3 star0 star12
0
1
0 10 12 10
1
end_operator
begin_operator
turn_to satellite3 star0 star4
0
1
0 10 13 10
1
end_operator
begin_operator
turn_to satellite3 star0 star7
0
1
0 10 14 10
1
end_operator
begin_operator
turn_to satellite3 star0 star9
0
1
0 10 15 10
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation1
0
1
0 10 0 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation10
0
1
0 10 1 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation13
0
1
0 10 2 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation14
0
1
0 10 3 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation2
0
1
0 10 4 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation3
0
1
0 10 5 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation5
0
1
0 10 6 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation6
0
1
0 10 7 11
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation8
0
1
0 10 8 11
1
end_operator
begin_operator
turn_to satellite3 star11 planet15
0
1
0 10 9 11
1
end_operator
begin_operator
turn_to satellite3 star11 star0
0
1
0 10 10 11
1
end_operator
begin_operator
turn_to satellite3 star11 star12
0
1
0 10 12 11
1
end_operator
begin_operator
turn_to satellite3 star11 star4
0
1
0 10 13 11
1
end_operator
begin_operator
turn_to satellite3 star11 star7
0
1
0 10 14 11
1
end_operator
begin_operator
turn_to satellite3 star11 star9
0
1
0 10 15 11
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation1
0
1
0 10 0 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation10
0
1
0 10 1 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation13
0
1
0 10 2 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation14
0
1
0 10 3 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation2
0
1
0 10 4 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation3
0
1
0 10 5 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation5
0
1
0 10 6 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation6
0
1
0 10 7 12
1
end_operator
begin_operator
turn_to satellite3 star12 groundstation8
0
1
0 10 8 12
1
end_operator
begin_operator
turn_to satellite3 star12 planet15
0
1
0 10 9 12
1
end_operator
begin_operator
turn_to satellite3 star12 star0
0
1
0 10 10 12
1
end_operator
begin_operator
turn_to satellite3 star12 star11
0
1
0 10 11 12
1
end_operator
begin_operator
turn_to satellite3 star12 star4
0
1
0 10 13 12
1
end_operator
begin_operator
turn_to satellite3 star12 star7
0
1
0 10 14 12
1
end_operator
begin_operator
turn_to satellite3 star12 star9
0
1
0 10 15 12
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation1
0
1
0 10 0 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation10
0
1
0 10 1 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation13
0
1
0 10 2 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation14
0
1
0 10 3 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation2
0
1
0 10 4 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation3
0
1
0 10 5 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation5
0
1
0 10 6 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation6
0
1
0 10 7 13
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation8
0
1
0 10 8 13
1
end_operator
begin_operator
turn_to satellite3 star4 planet15
0
1
0 10 9 13
1
end_operator
begin_operator
turn_to satellite3 star4 star0
0
1
0 10 10 13
1
end_operator
begin_operator
turn_to satellite3 star4 star11
0
1
0 10 11 13
1
end_operator
begin_operator
turn_to satellite3 star4 star12
0
1
0 10 12 13
1
end_operator
begin_operator
turn_to satellite3 star4 star7
0
1
0 10 14 13
1
end_operator
begin_operator
turn_to satellite3 star4 star9
0
1
0 10 15 13
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation1
0
1
0 10 0 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation10
0
1
0 10 1 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation13
0
1
0 10 2 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation14
0
1
0 10 3 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation2
0
1
0 10 4 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation3
0
1
0 10 5 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation5
0
1
0 10 6 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation6
0
1
0 10 7 14
1
end_operator
begin_operator
turn_to satellite3 star7 groundstation8
0
1
0 10 8 14
1
end_operator
begin_operator
turn_to satellite3 star7 planet15
0
1
0 10 9 14
1
end_operator
begin_operator
turn_to satellite3 star7 star0
0
1
0 10 10 14
1
end_operator
begin_operator
turn_to satellite3 star7 star11
0
1
0 10 11 14
1
end_operator
begin_operator
turn_to satellite3 star7 star12
0
1
0 10 12 14
1
end_operator
begin_operator
turn_to satellite3 star7 star4
0
1
0 10 13 14
1
end_operator
begin_operator
turn_to satellite3 star7 star9
0
1
0 10 15 14
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation1
0
1
0 10 0 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation10
0
1
0 10 1 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation13
0
1
0 10 2 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation14
0
1
0 10 3 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation2
0
1
0 10 4 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation3
0
1
0 10 5 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation5
0
1
0 10 6 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation6
0
1
0 10 7 15
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation8
0
1
0 10 8 15
1
end_operator
begin_operator
turn_to satellite3 star9 planet15
0
1
0 10 9 15
1
end_operator
begin_operator
turn_to satellite3 star9 star0
0
1
0 10 10 15
1
end_operator
begin_operator
turn_to satellite3 star9 star11
0
1
0 10 11 15
1
end_operator
begin_operator
turn_to satellite3 star9 star12
0
1
0 10 12 15
1
end_operator
begin_operator
turn_to satellite3 star9 star4
0
1
0 10 13 15
1
end_operator
begin_operator
turn_to satellite3 star9 star7
0
1
0 10 14 15
1
end_operator
0
