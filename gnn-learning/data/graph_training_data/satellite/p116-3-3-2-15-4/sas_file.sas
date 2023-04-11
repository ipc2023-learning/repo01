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
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
end_variable
begin_variable
var3
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var5
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var6
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var7
-1
19
Atom pointing(satellite2, groundstation10)
Atom pointing(satellite2, groundstation14)
Atom pointing(satellite2, groundstation3)
Atom pointing(satellite2, groundstation7)
Atom pointing(satellite2, groundstation8)
Atom pointing(satellite2, phenomenon15)
Atom pointing(satellite2, phenomenon17)
Atom pointing(satellite2, planet16)
Atom pointing(satellite2, star0)
Atom pointing(satellite2, star1)
Atom pointing(satellite2, star11)
Atom pointing(satellite2, star12)
Atom pointing(satellite2, star13)
Atom pointing(satellite2, star18)
Atom pointing(satellite2, star2)
Atom pointing(satellite2, star4)
Atom pointing(satellite2, star5)
Atom pointing(satellite2, star6)
Atom pointing(satellite2, star9)
end_variable
begin_variable
var8
-1
19
Atom pointing(satellite1, groundstation10)
Atom pointing(satellite1, groundstation14)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation7)
Atom pointing(satellite1, groundstation8)
Atom pointing(satellite1, phenomenon15)
Atom pointing(satellite1, phenomenon17)
Atom pointing(satellite1, planet16)
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star11)
Atom pointing(satellite1, star12)
Atom pointing(satellite1, star13)
Atom pointing(satellite1, star18)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star4)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star6)
Atom pointing(satellite1, star9)
end_variable
begin_variable
var9
-1
19
Atom pointing(satellite0, groundstation10)
Atom pointing(satellite0, groundstation14)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation7)
Atom pointing(satellite0, groundstation8)
Atom pointing(satellite0, phenomenon15)
Atom pointing(satellite0, phenomenon17)
Atom pointing(satellite0, planet16)
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star11)
Atom pointing(satellite0, star12)
Atom pointing(satellite0, star13)
Atom pointing(satellite0, star18)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star4)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star6)
Atom pointing(satellite0, star9)
end_variable
begin_variable
var10
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var11
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var12
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var13
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var14
-1
2
Atom have_image(star18, infrared0)
NegatedAtom have_image(star18, infrared0)
end_variable
begin_variable
var15
-1
2
Atom have_image(planet16, infrared0)
NegatedAtom have_image(planet16, infrared0)
end_variable
begin_variable
var16
-1
2
Atom have_image(phenomenon17, infrared0)
NegatedAtom have_image(phenomenon17, infrared0)
end_variable
begin_variable
var17
-1
2
Atom have_image(phenomenon15, infrared0)
NegatedAtom have_image(phenomenon15, infrared0)
end_variable
0
begin_state
1
1
0
0
1
0
1
16
11
13
1
1
1
1
1
1
1
1
end_state
begin_goal
5
7 14
14 0
15 0
16 0
17 0
end_goal
1061
begin_operator
calibrate satellite0 instrument0 star4
2
9 15
6 0
1
0 13 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 star5
2
9 16
6 0
1
0 13 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation3
2
8 2
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation8
2
8 4
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star11
2
8 10
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star2
2
8 14
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument2 groundstation10
2
7 0
0 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument2 star1
2
7 9
0 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 groundstation14
2
7 1
1 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 groundstation7
2
7 3
1 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 star9
2
7 18
1 0
1
0 10 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 5 -1 0
0 6 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite1
0
2
0 3 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite2
0
2
0 2 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite2
0
2
0 2 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 13 -1 1
0 5 0 1
0 6 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 12 -1 1
0 3 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite2
0
3
0 11 -1 1
0 2 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite2
0
3
0 10 -1 1
0 2 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon15 instrument0 infrared0
3
13 0
9 5
6 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon17 instrument0 infrared0
3
13 0
9 6
6 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 planet16 instrument0 infrared0
3
13 0
9 7
6 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star18 instrument0 infrared0
3
13 0
9 13
6 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon15 instrument1 infrared0
3
12 0
8 5
4 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon17 instrument1 infrared0
3
12 0
8 6
4 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 planet16 instrument1 infrared0
3
12 0
8 7
4 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 star18 instrument1 infrared0
3
12 0
8 13
4 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon15 instrument2 infrared0
3
11 0
7 5
0 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon15 instrument3 infrared0
3
10 0
7 5
1 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon17 instrument2 infrared0
3
11 0
7 6
0 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon17 instrument3 infrared0
3
10 0
7 6
1 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite2 planet16 instrument2 infrared0
3
11 0
7 7
0 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 planet16 instrument3 infrared0
3
10 0
7 7
1 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 star18 instrument2 infrared0
3
11 0
7 13
0 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite2 star18 instrument3 infrared0
3
10 0
7 13
1 0
1
0 14 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation14
0
1
0 9 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation3
0
1
0 9 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation7
0
1
0 9 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 groundstation8
0
1
0 9 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 phenomenon15
0
1
0 9 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 phenomenon17
0
1
0 9 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 planet16
0
1
0 9 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star0
0
1
0 9 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star1
0
1
0 9 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star11
0
1
0 9 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star12
0
1
0 9 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star13
0
1
0 9 12 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star18
0
1
0 9 13 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star2
0
1
0 9 14 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star4
0
1
0 9 15 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star5
0
1
0 9 16 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star6
0
1
0 9 17 0
1
end_operator
begin_operator
turn_to satellite0 groundstation10 star9
0
1
0 9 18 0
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation10
0
1
0 9 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation3
0
1
0 9 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation7
0
1
0 9 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation8
0
1
0 9 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 phenomenon15
0
1
0 9 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 phenomenon17
0
1
0 9 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 planet16
0
1
0 9 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star0
0
1
0 9 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star1
0
1
0 9 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star11
0
1
0 9 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star12
0
1
0 9 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star13
0
1
0 9 12 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star18
0
1
0 9 13 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star2
0
1
0 9 14 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star4
0
1
0 9 15 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star5
0
1
0 9 16 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star6
0
1
0 9 17 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star9
0
1
0 9 18 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation10
0
1
0 9 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation14
0
1
0 9 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation7
0
1
0 9 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation8
0
1
0 9 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon15
0
1
0 9 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon17
0
1
0 9 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet16
0
1
0 9 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star0
0
1
0 9 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star1
0
1
0 9 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star11
0
1
0 9 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star12
0
1
0 9 11 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star13
0
1
0 9 12 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star18
0
1
0 9 13 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star2
0
1
0 9 14 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star4
0
1
0 9 15 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star5
0
1
0 9 16 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star6
0
1
0 9 17 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star9
0
1
0 9 18 2
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation10
0
1
0 9 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation14
0
1
0 9 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation3
0
1
0 9 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation8
0
1
0 9 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 phenomenon15
0
1
0 9 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 phenomenon17
0
1
0 9 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 planet16
0
1
0 9 7 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star0
0
1
0 9 8 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star1
0
1
0 9 9 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star11
0
1
0 9 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star12
0
1
0 9 11 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star13
0
1
0 9 12 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star18
0
1
0 9 13 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star2
0
1
0 9 14 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star4
0
1
0 9 15 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star5
0
1
0 9 16 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star6
0
1
0 9 17 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star9
0
1
0 9 18 3
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation10
0
1
0 9 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation14
0
1
0 9 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation3
0
1
0 9 2 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation7
0
1
0 9 3 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 phenomenon15
0
1
0 9 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 phenomenon17
0
1
0 9 6 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 planet16
0
1
0 9 7 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star0
0
1
0 9 8 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star1
0
1
0 9 9 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star11
0
1
0 9 10 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star12
0
1
0 9 11 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star13
0
1
0 9 12 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star18
0
1
0 9 13 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star2
0
1
0 9 14 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star4
0
1
0 9 15 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star5
0
1
0 9 16 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star6
0
1
0 9 17 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star9
0
1
0 9 18 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 groundstation10
0
1
0 9 0 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 groundstation14
0
1
0 9 1 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 groundstation3
0
1
0 9 2 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 groundstation7
0
1
0 9 3 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 groundstation8
0
1
0 9 4 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 phenomenon17
0
1
0 9 6 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 planet16
0
1
0 9 7 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star0
0
1
0 9 8 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star1
0
1
0 9 9 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star11
0
1
0 9 10 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star12
0
1
0 9 11 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star13
0
1
0 9 12 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star18
0
1
0 9 13 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star2
0
1
0 9 14 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star4
0
1
0 9 15 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star5
0
1
0 9 16 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star6
0
1
0 9 17 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon15 star9
0
1
0 9 18 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 groundstation10
0
1
0 9 0 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 groundstation14
0
1
0 9 1 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 groundstation3
0
1
0 9 2 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 groundstation7
0
1
0 9 3 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 groundstation8
0
1
0 9 4 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 phenomenon15
0
1
0 9 5 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 planet16
0
1
0 9 7 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star0
0
1
0 9 8 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star1
0
1
0 9 9 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star11
0
1
0 9 10 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star12
0
1
0 9 11 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star13
0
1
0 9 12 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star18
0
1
0 9 13 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star2
0
1
0 9 14 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star4
0
1
0 9 15 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star5
0
1
0 9 16 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star6
0
1
0 9 17 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon17 star9
0
1
0 9 18 6
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation10
0
1
0 9 0 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation14
0
1
0 9 1 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation3
0
1
0 9 2 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation7
0
1
0 9 3 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation8
0
1
0 9 4 7
1
end_operator
begin_operator
turn_to satellite0 planet16 phenomenon15
0
1
0 9 5 7
1
end_operator
begin_operator
turn_to satellite0 planet16 phenomenon17
0
1
0 9 6 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star0
0
1
0 9 8 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star1
0
1
0 9 9 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star11
0
1
0 9 10 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star12
0
1
0 9 11 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star13
0
1
0 9 12 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star18
0
1
0 9 13 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star2
0
1
0 9 14 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star4
0
1
0 9 15 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star5
0
1
0 9 16 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star6
0
1
0 9 17 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star9
0
1
0 9 18 7
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation10
0
1
0 9 0 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation14
0
1
0 9 1 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation3
0
1
0 9 2 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation7
0
1
0 9 3 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation8
0
1
0 9 4 8
1
end_operator
begin_operator
turn_to satellite0 star0 phenomenon15
0
1
0 9 5 8
1
end_operator
begin_operator
turn_to satellite0 star0 phenomenon17
0
1
0 9 6 8
1
end_operator
begin_operator
turn_to satellite0 star0 planet16
0
1
0 9 7 8
1
end_operator
begin_operator
turn_to satellite0 star0 star1
0
1
0 9 9 8
1
end_operator
begin_operator
turn_to satellite0 star0 star11
0
1
0 9 10 8
1
end_operator
begin_operator
turn_to satellite0 star0 star12
0
1
0 9 11 8
1
end_operator
begin_operator
turn_to satellite0 star0 star13
0
1
0 9 12 8
1
end_operator
begin_operator
turn_to satellite0 star0 star18
0
1
0 9 13 8
1
end_operator
begin_operator
turn_to satellite0 star0 star2
0
1
0 9 14 8
1
end_operator
begin_operator
turn_to satellite0 star0 star4
0
1
0 9 15 8
1
end_operator
begin_operator
turn_to satellite0 star0 star5
0
1
0 9 16 8
1
end_operator
begin_operator
turn_to satellite0 star0 star6
0
1
0 9 17 8
1
end_operator
begin_operator
turn_to satellite0 star0 star9
0
1
0 9 18 8
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation10
0
1
0 9 0 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation14
0
1
0 9 1 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation3
0
1
0 9 2 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation7
0
1
0 9 3 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation8
0
1
0 9 4 9
1
end_operator
begin_operator
turn_to satellite0 star1 phenomenon15
0
1
0 9 5 9
1
end_operator
begin_operator
turn_to satellite0 star1 phenomenon17
0
1
0 9 6 9
1
end_operator
begin_operator
turn_to satellite0 star1 planet16
0
1
0 9 7 9
1
end_operator
begin_operator
turn_to satellite0 star1 star0
0
1
0 9 8 9
1
end_operator
begin_operator
turn_to satellite0 star1 star11
0
1
0 9 10 9
1
end_operator
begin_operator
turn_to satellite0 star1 star12
0
1
0 9 11 9
1
end_operator
begin_operator
turn_to satellite0 star1 star13
0
1
0 9 12 9
1
end_operator
begin_operator
turn_to satellite0 star1 star18
0
1
0 9 13 9
1
end_operator
begin_operator
turn_to satellite0 star1 star2
0
1
0 9 14 9
1
end_operator
begin_operator
turn_to satellite0 star1 star4
0
1
0 9 15 9
1
end_operator
begin_operator
turn_to satellite0 star1 star5
0
1
0 9 16 9
1
end_operator
begin_operator
turn_to satellite0 star1 star6
0
1
0 9 17 9
1
end_operator
begin_operator
turn_to satellite0 star1 star9
0
1
0 9 18 9
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation10
0
1
0 9 0 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation14
0
1
0 9 1 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation3
0
1
0 9 2 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation7
0
1
0 9 3 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation8
0
1
0 9 4 10
1
end_operator
begin_operator
turn_to satellite0 star11 phenomenon15
0
1
0 9 5 10
1
end_operator
begin_operator
turn_to satellite0 star11 phenomenon17
0
1
0 9 6 10
1
end_operator
begin_operator
turn_to satellite0 star11 planet16
0
1
0 9 7 10
1
end_operator
begin_operator
turn_to satellite0 star11 star0
0
1
0 9 8 10
1
end_operator
begin_operator
turn_to satellite0 star11 star1
0
1
0 9 9 10
1
end_operator
begin_operator
turn_to satellite0 star11 star12
0
1
0 9 11 10
1
end_operator
begin_operator
turn_to satellite0 star11 star13
0
1
0 9 12 10
1
end_operator
begin_operator
turn_to satellite0 star11 star18
0
1
0 9 13 10
1
end_operator
begin_operator
turn_to satellite0 star11 star2
0
1
0 9 14 10
1
end_operator
begin_operator
turn_to satellite0 star11 star4
0
1
0 9 15 10
1
end_operator
begin_operator
turn_to satellite0 star11 star5
0
1
0 9 16 10
1
end_operator
begin_operator
turn_to satellite0 star11 star6
0
1
0 9 17 10
1
end_operator
begin_operator
turn_to satellite0 star11 star9
0
1
0 9 18 10
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation10
0
1
0 9 0 11
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation14
0
1
0 9 1 11
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation3
0
1
0 9 2 11
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation7
0
1
0 9 3 11
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation8
0
1
0 9 4 11
1
end_operator
begin_operator
turn_to satellite0 star12 phenomenon15
0
1
0 9 5 11
1
end_operator
begin_operator
turn_to satellite0 star12 phenomenon17
0
1
0 9 6 11
1
end_operator
begin_operator
turn_to satellite0 star12 planet16
0
1
0 9 7 11
1
end_operator
begin_operator
turn_to satellite0 star12 star0
0
1
0 9 8 11
1
end_operator
begin_operator
turn_to satellite0 star12 star1
0
1
0 9 9 11
1
end_operator
begin_operator
turn_to satellite0 star12 star11
0
1
0 9 10 11
1
end_operator
begin_operator
turn_to satellite0 star12 star13
0
1
0 9 12 11
1
end_operator
begin_operator
turn_to satellite0 star12 star18
0
1
0 9 13 11
1
end_operator
begin_operator
turn_to satellite0 star12 star2
0
1
0 9 14 11
1
end_operator
begin_operator
turn_to satellite0 star12 star4
0
1
0 9 15 11
1
end_operator
begin_operator
turn_to satellite0 star12 star5
0
1
0 9 16 11
1
end_operator
begin_operator
turn_to satellite0 star12 star6
0
1
0 9 17 11
1
end_operator
begin_operator
turn_to satellite0 star12 star9
0
1
0 9 18 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation10
0
1
0 9 0 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation14
0
1
0 9 1 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation3
0
1
0 9 2 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation7
0
1
0 9 3 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation8
0
1
0 9 4 12
1
end_operator
begin_operator
turn_to satellite0 star13 phenomenon15
0
1
0 9 5 12
1
end_operator
begin_operator
turn_to satellite0 star13 phenomenon17
0
1
0 9 6 12
1
end_operator
begin_operator
turn_to satellite0 star13 planet16
0
1
0 9 7 12
1
end_operator
begin_operator
turn_to satellite0 star13 star0
0
1
0 9 8 12
1
end_operator
begin_operator
turn_to satellite0 star13 star1
0
1
0 9 9 12
1
end_operator
begin_operator
turn_to satellite0 star13 star11
0
1
0 9 10 12
1
end_operator
begin_operator
turn_to satellite0 star13 star12
0
1
0 9 11 12
1
end_operator
begin_operator
turn_to satellite0 star13 star18
0
1
0 9 13 12
1
end_operator
begin_operator
turn_to satellite0 star13 star2
0
1
0 9 14 12
1
end_operator
begin_operator
turn_to satellite0 star13 star4
0
1
0 9 15 12
1
end_operator
begin_operator
turn_to satellite0 star13 star5
0
1
0 9 16 12
1
end_operator
begin_operator
turn_to satellite0 star13 star6
0
1
0 9 17 12
1
end_operator
begin_operator
turn_to satellite0 star13 star9
0
1
0 9 18 12
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation10
0
1
0 9 0 13
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation14
0
1
0 9 1 13
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation3
0
1
0 9 2 13
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation7
0
1
0 9 3 13
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation8
0
1
0 9 4 13
1
end_operator
begin_operator
turn_to satellite0 star18 phenomenon15
0
1
0 9 5 13
1
end_operator
begin_operator
turn_to satellite0 star18 phenomenon17
0
1
0 9 6 13
1
end_operator
begin_operator
turn_to satellite0 star18 planet16
0
1
0 9 7 13
1
end_operator
begin_operator
turn_to satellite0 star18 star0
0
1
0 9 8 13
1
end_operator
begin_operator
turn_to satellite0 star18 star1
0
1
0 9 9 13
1
end_operator
begin_operator
turn_to satellite0 star18 star11
0
1
0 9 10 13
1
end_operator
begin_operator
turn_to satellite0 star18 star12
0
1
0 9 11 13
1
end_operator
begin_operator
turn_to satellite0 star18 star13
0
1
0 9 12 13
1
end_operator
begin_operator
turn_to satellite0 star18 star2
0
1
0 9 14 13
1
end_operator
begin_operator
turn_to satellite0 star18 star4
0
1
0 9 15 13
1
end_operator
begin_operator
turn_to satellite0 star18 star5
0
1
0 9 16 13
1
end_operator
begin_operator
turn_to satellite0 star18 star6
0
1
0 9 17 13
1
end_operator
begin_operator
turn_to satellite0 star18 star9
0
1
0 9 18 13
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation10
0
1
0 9 0 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation14
0
1
0 9 1 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation3
0
1
0 9 2 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation7
0
1
0 9 3 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation8
0
1
0 9 4 14
1
end_operator
begin_operator
turn_to satellite0 star2 phenomenon15
0
1
0 9 5 14
1
end_operator
begin_operator
turn_to satellite0 star2 phenomenon17
0
1
0 9 6 14
1
end_operator
begin_operator
turn_to satellite0 star2 planet16
0
1
0 9 7 14
1
end_operator
begin_operator
turn_to satellite0 star2 star0
0
1
0 9 8 14
1
end_operator
begin_operator
turn_to satellite0 star2 star1
0
1
0 9 9 14
1
end_operator
begin_operator
turn_to satellite0 star2 star11
0
1
0 9 10 14
1
end_operator
begin_operator
turn_to satellite0 star2 star12
0
1
0 9 11 14
1
end_operator
begin_operator
turn_to satellite0 star2 star13
0
1
0 9 12 14
1
end_operator
begin_operator
turn_to satellite0 star2 star18
0
1
0 9 13 14
1
end_operator
begin_operator
turn_to satellite0 star2 star4
0
1
0 9 15 14
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 9 16 14
1
end_operator
begin_operator
turn_to satellite0 star2 star6
0
1
0 9 17 14
1
end_operator
begin_operator
turn_to satellite0 star2 star9
0
1
0 9 18 14
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation10
0
1
0 9 0 15
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation14
0
1
0 9 1 15
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation3
0
1
0 9 2 15
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation7
0
1
0 9 3 15
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation8
0
1
0 9 4 15
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon15
0
1
0 9 5 15
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon17
0
1
0 9 6 15
1
end_operator
begin_operator
turn_to satellite0 star4 planet16
0
1
0 9 7 15
1
end_operator
begin_operator
turn_to satellite0 star4 star0
0
1
0 9 8 15
1
end_operator
begin_operator
turn_to satellite0 star4 star1
0
1
0 9 9 15
1
end_operator
begin_operator
turn_to satellite0 star4 star11
0
1
0 9 10 15
1
end_operator
begin_operator
turn_to satellite0 star4 star12
0
1
0 9 11 15
1
end_operator
begin_operator
turn_to satellite0 star4 star13
0
1
0 9 12 15
1
end_operator
begin_operator
turn_to satellite0 star4 star18
0
1
0 9 13 15
1
end_operator
begin_operator
turn_to satellite0 star4 star2
0
1
0 9 14 15
1
end_operator
begin_operator
turn_to satellite0 star4 star5
0
1
0 9 16 15
1
end_operator
begin_operator
turn_to satellite0 star4 star6
0
1
0 9 17 15
1
end_operator
begin_operator
turn_to satellite0 star4 star9
0
1
0 9 18 15
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation10
0
1
0 9 0 16
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation14
0
1
0 9 1 16
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation3
0
1
0 9 2 16
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation7
0
1
0 9 3 16
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation8
0
1
0 9 4 16
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon15
0
1
0 9 5 16
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon17
0
1
0 9 6 16
1
end_operator
begin_operator
turn_to satellite0 star5 planet16
0
1
0 9 7 16
1
end_operator
begin_operator
turn_to satellite0 star5 star0
0
1
0 9 8 16
1
end_operator
begin_operator
turn_to satellite0 star5 star1
0
1
0 9 9 16
1
end_operator
begin_operator
turn_to satellite0 star5 star11
0
1
0 9 10 16
1
end_operator
begin_operator
turn_to satellite0 star5 star12
0
1
0 9 11 16
1
end_operator
begin_operator
turn_to satellite0 star5 star13
0
1
0 9 12 16
1
end_operator
begin_operator
turn_to satellite0 star5 star18
0
1
0 9 13 16
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 9 14 16
1
end_operator
begin_operator
turn_to satellite0 star5 star4
0
1
0 9 15 16
1
end_operator
begin_operator
turn_to satellite0 star5 star6
0
1
0 9 17 16
1
end_operator
begin_operator
turn_to satellite0 star5 star9
0
1
0 9 18 16
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation10
0
1
0 9 0 17
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation14
0
1
0 9 1 17
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation3
0
1
0 9 2 17
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation7
0
1
0 9 3 17
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation8
0
1
0 9 4 17
1
end_operator
begin_operator
turn_to satellite0 star6 phenomenon15
0
1
0 9 5 17
1
end_operator
begin_operator
turn_to satellite0 star6 phenomenon17
0
1
0 9 6 17
1
end_operator
begin_operator
turn_to satellite0 star6 planet16
0
1
0 9 7 17
1
end_operator
begin_operator
turn_to satellite0 star6 star0
0
1
0 9 8 17
1
end_operator
begin_operator
turn_to satellite0 star6 star1
0
1
0 9 9 17
1
end_operator
begin_operator
turn_to satellite0 star6 star11
0
1
0 9 10 17
1
end_operator
begin_operator
turn_to satellite0 star6 star12
0
1
0 9 11 17
1
end_operator
begin_operator
turn_to satellite0 star6 star13
0
1
0 9 12 17
1
end_operator
begin_operator
turn_to satellite0 star6 star18
0
1
0 9 13 17
1
end_operator
begin_operator
turn_to satellite0 star6 star2
0
1
0 9 14 17
1
end_operator
begin_operator
turn_to satellite0 star6 star4
0
1
0 9 15 17
1
end_operator
begin_operator
turn_to satellite0 star6 star5
0
1
0 9 16 17
1
end_operator
begin_operator
turn_to satellite0 star6 star9
0
1
0 9 18 17
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation10
0
1
0 9 0 18
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation14
0
1
0 9 1 18
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation3
0
1
0 9 2 18
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation7
0
1
0 9 3 18
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation8
0
1
0 9 4 18
1
end_operator
begin_operator
turn_to satellite0 star9 phenomenon15
0
1
0 9 5 18
1
end_operator
begin_operator
turn_to satellite0 star9 phenomenon17
0
1
0 9 6 18
1
end_operator
begin_operator
turn_to satellite0 star9 planet16
0
1
0 9 7 18
1
end_operator
begin_operator
turn_to satellite0 star9 star0
0
1
0 9 8 18
1
end_operator
begin_operator
turn_to satellite0 star9 star1
0
1
0 9 9 18
1
end_operator
begin_operator
turn_to satellite0 star9 star11
0
1
0 9 10 18
1
end_operator
begin_operator
turn_to satellite0 star9 star12
0
1
0 9 11 18
1
end_operator
begin_operator
turn_to satellite0 star9 star13
0
1
0 9 12 18
1
end_operator
begin_operator
turn_to satellite0 star9 star18
0
1
0 9 13 18
1
end_operator
begin_operator
turn_to satellite0 star9 star2
0
1
0 9 14 18
1
end_operator
begin_operator
turn_to satellite0 star9 star4
0
1
0 9 15 18
1
end_operator
begin_operator
turn_to satellite0 star9 star5
0
1
0 9 16 18
1
end_operator
begin_operator
turn_to satellite0 star9 star6
0
1
0 9 17 18
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation14
0
1
0 8 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation3
0
1
0 8 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation7
0
1
0 8 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 groundstation8
0
1
0 8 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 phenomenon15
0
1
0 8 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 phenomenon17
0
1
0 8 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 planet16
0
1
0 8 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star0
0
1
0 8 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star1
0
1
0 8 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star11
0
1
0 8 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star12
0
1
0 8 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star13
0
1
0 8 12 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star18
0
1
0 8 13 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star2
0
1
0 8 14 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star4
0
1
0 8 15 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star5
0
1
0 8 16 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star6
0
1
0 8 17 0
1
end_operator
begin_operator
turn_to satellite1 groundstation10 star9
0
1
0 8 18 0
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation10
0
1
0 8 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation3
0
1
0 8 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation7
0
1
0 8 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation8
0
1
0 8 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 phenomenon15
0
1
0 8 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 phenomenon17
0
1
0 8 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 planet16
0
1
0 8 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star0
0
1
0 8 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star1
0
1
0 8 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star11
0
1
0 8 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star12
0
1
0 8 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star13
0
1
0 8 12 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star18
0
1
0 8 13 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star2
0
1
0 8 14 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star4
0
1
0 8 15 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star5
0
1
0 8 16 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star6
0
1
0 8 17 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star9
0
1
0 8 18 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation10
0
1
0 8 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation14
0
1
0 8 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation7
0
1
0 8 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation8
0
1
0 8 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon15
0
1
0 8 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon17
0
1
0 8 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet16
0
1
0 8 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star0
0
1
0 8 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star1
0
1
0 8 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star11
0
1
0 8 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star12
0
1
0 8 11 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star13
0
1
0 8 12 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star18
0
1
0 8 13 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star2
0
1
0 8 14 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star4
0
1
0 8 15 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star5
0
1
0 8 16 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star6
0
1
0 8 17 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star9
0
1
0 8 18 2
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation10
0
1
0 8 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation14
0
1
0 8 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation3
0
1
0 8 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation8
0
1
0 8 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 phenomenon15
0
1
0 8 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 phenomenon17
0
1
0 8 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 planet16
0
1
0 8 7 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star0
0
1
0 8 8 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star1
0
1
0 8 9 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star11
0
1
0 8 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star12
0
1
0 8 11 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star13
0
1
0 8 12 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star18
0
1
0 8 13 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star2
0
1
0 8 14 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star4
0
1
0 8 15 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star5
0
1
0 8 16 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star6
0
1
0 8 17 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star9
0
1
0 8 18 3
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation10
0
1
0 8 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation14
0
1
0 8 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation3
0
1
0 8 2 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation7
0
1
0 8 3 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 phenomenon15
0
1
0 8 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 phenomenon17
0
1
0 8 6 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 planet16
0
1
0 8 7 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star0
0
1
0 8 8 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star1
0
1
0 8 9 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star11
0
1
0 8 10 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star12
0
1
0 8 11 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star13
0
1
0 8 12 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star18
0
1
0 8 13 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star2
0
1
0 8 14 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star4
0
1
0 8 15 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star5
0
1
0 8 16 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star6
0
1
0 8 17 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star9
0
1
0 8 18 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 groundstation10
0
1
0 8 0 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 groundstation14
0
1
0 8 1 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 groundstation3
0
1
0 8 2 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 groundstation7
0
1
0 8 3 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 groundstation8
0
1
0 8 4 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 phenomenon17
0
1
0 8 6 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 planet16
0
1
0 8 7 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star0
0
1
0 8 8 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star1
0
1
0 8 9 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star11
0
1
0 8 10 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star12
0
1
0 8 11 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star13
0
1
0 8 12 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star18
0
1
0 8 13 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star2
0
1
0 8 14 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star4
0
1
0 8 15 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star5
0
1
0 8 16 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star6
0
1
0 8 17 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon15 star9
0
1
0 8 18 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 groundstation10
0
1
0 8 0 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 groundstation14
0
1
0 8 1 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 groundstation3
0
1
0 8 2 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 groundstation7
0
1
0 8 3 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 groundstation8
0
1
0 8 4 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 phenomenon15
0
1
0 8 5 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 planet16
0
1
0 8 7 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star0
0
1
0 8 8 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star1
0
1
0 8 9 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star11
0
1
0 8 10 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star12
0
1
0 8 11 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star13
0
1
0 8 12 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star18
0
1
0 8 13 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star2
0
1
0 8 14 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star4
0
1
0 8 15 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star5
0
1
0 8 16 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star6
0
1
0 8 17 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon17 star9
0
1
0 8 18 6
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation10
0
1
0 8 0 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation14
0
1
0 8 1 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation3
0
1
0 8 2 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation7
0
1
0 8 3 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation8
0
1
0 8 4 7
1
end_operator
begin_operator
turn_to satellite1 planet16 phenomenon15
0
1
0 8 5 7
1
end_operator
begin_operator
turn_to satellite1 planet16 phenomenon17
0
1
0 8 6 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star0
0
1
0 8 8 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star1
0
1
0 8 9 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star11
0
1
0 8 10 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star12
0
1
0 8 11 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star13
0
1
0 8 12 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star18
0
1
0 8 13 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star2
0
1
0 8 14 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star4
0
1
0 8 15 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star5
0
1
0 8 16 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star6
0
1
0 8 17 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star9
0
1
0 8 18 7
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation10
0
1
0 8 0 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation14
0
1
0 8 1 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation3
0
1
0 8 2 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation7
0
1
0 8 3 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation8
0
1
0 8 4 8
1
end_operator
begin_operator
turn_to satellite1 star0 phenomenon15
0
1
0 8 5 8
1
end_operator
begin_operator
turn_to satellite1 star0 phenomenon17
0
1
0 8 6 8
1
end_operator
begin_operator
turn_to satellite1 star0 planet16
0
1
0 8 7 8
1
end_operator
begin_operator
turn_to satellite1 star0 star1
0
1
0 8 9 8
1
end_operator
begin_operator
turn_to satellite1 star0 star11
0
1
0 8 10 8
1
end_operator
begin_operator
turn_to satellite1 star0 star12
0
1
0 8 11 8
1
end_operator
begin_operator
turn_to satellite1 star0 star13
0
1
0 8 12 8
1
end_operator
begin_operator
turn_to satellite1 star0 star18
0
1
0 8 13 8
1
end_operator
begin_operator
turn_to satellite1 star0 star2
0
1
0 8 14 8
1
end_operator
begin_operator
turn_to satellite1 star0 star4
0
1
0 8 15 8
1
end_operator
begin_operator
turn_to satellite1 star0 star5
0
1
0 8 16 8
1
end_operator
begin_operator
turn_to satellite1 star0 star6
0
1
0 8 17 8
1
end_operator
begin_operator
turn_to satellite1 star0 star9
0
1
0 8 18 8
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation10
0
1
0 8 0 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation14
0
1
0 8 1 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation3
0
1
0 8 2 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation7
0
1
0 8 3 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation8
0
1
0 8 4 9
1
end_operator
begin_operator
turn_to satellite1 star1 phenomenon15
0
1
0 8 5 9
1
end_operator
begin_operator
turn_to satellite1 star1 phenomenon17
0
1
0 8 6 9
1
end_operator
begin_operator
turn_to satellite1 star1 planet16
0
1
0 8 7 9
1
end_operator
begin_operator
turn_to satellite1 star1 star0
0
1
0 8 8 9
1
end_operator
begin_operator
turn_to satellite1 star1 star11
0
1
0 8 10 9
1
end_operator
begin_operator
turn_to satellite1 star1 star12
0
1
0 8 11 9
1
end_operator
begin_operator
turn_to satellite1 star1 star13
0
1
0 8 12 9
1
end_operator
begin_operator
turn_to satellite1 star1 star18
0
1
0 8 13 9
1
end_operator
begin_operator
turn_to satellite1 star1 star2
0
1
0 8 14 9
1
end_operator
begin_operator
turn_to satellite1 star1 star4
0
1
0 8 15 9
1
end_operator
begin_operator
turn_to satellite1 star1 star5
0
1
0 8 16 9
1
end_operator
begin_operator
turn_to satellite1 star1 star6
0
1
0 8 17 9
1
end_operator
begin_operator
turn_to satellite1 star1 star9
0
1
0 8 18 9
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation10
0
1
0 8 0 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation14
0
1
0 8 1 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation3
0
1
0 8 2 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation7
0
1
0 8 3 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation8
0
1
0 8 4 10
1
end_operator
begin_operator
turn_to satellite1 star11 phenomenon15
0
1
0 8 5 10
1
end_operator
begin_operator
turn_to satellite1 star11 phenomenon17
0
1
0 8 6 10
1
end_operator
begin_operator
turn_to satellite1 star11 planet16
0
1
0 8 7 10
1
end_operator
begin_operator
turn_to satellite1 star11 star0
0
1
0 8 8 10
1
end_operator
begin_operator
turn_to satellite1 star11 star1
0
1
0 8 9 10
1
end_operator
begin_operator
turn_to satellite1 star11 star12
0
1
0 8 11 10
1
end_operator
begin_operator
turn_to satellite1 star11 star13
0
1
0 8 12 10
1
end_operator
begin_operator
turn_to satellite1 star11 star18
0
1
0 8 13 10
1
end_operator
begin_operator
turn_to satellite1 star11 star2
0
1
0 8 14 10
1
end_operator
begin_operator
turn_to satellite1 star11 star4
0
1
0 8 15 10
1
end_operator
begin_operator
turn_to satellite1 star11 star5
0
1
0 8 16 10
1
end_operator
begin_operator
turn_to satellite1 star11 star6
0
1
0 8 17 10
1
end_operator
begin_operator
turn_to satellite1 star11 star9
0
1
0 8 18 10
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation10
0
1
0 8 0 11
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation14
0
1
0 8 1 11
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation3
0
1
0 8 2 11
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation7
0
1
0 8 3 11
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation8
0
1
0 8 4 11
1
end_operator
begin_operator
turn_to satellite1 star12 phenomenon15
0
1
0 8 5 11
1
end_operator
begin_operator
turn_to satellite1 star12 phenomenon17
0
1
0 8 6 11
1
end_operator
begin_operator
turn_to satellite1 star12 planet16
0
1
0 8 7 11
1
end_operator
begin_operator
turn_to satellite1 star12 star0
0
1
0 8 8 11
1
end_operator
begin_operator
turn_to satellite1 star12 star1
0
1
0 8 9 11
1
end_operator
begin_operator
turn_to satellite1 star12 star11
0
1
0 8 10 11
1
end_operator
begin_operator
turn_to satellite1 star12 star13
0
1
0 8 12 11
1
end_operator
begin_operator
turn_to satellite1 star12 star18
0
1
0 8 13 11
1
end_operator
begin_operator
turn_to satellite1 star12 star2
0
1
0 8 14 11
1
end_operator
begin_operator
turn_to satellite1 star12 star4
0
1
0 8 15 11
1
end_operator
begin_operator
turn_to satellite1 star12 star5
0
1
0 8 16 11
1
end_operator
begin_operator
turn_to satellite1 star12 star6
0
1
0 8 17 11
1
end_operator
begin_operator
turn_to satellite1 star12 star9
0
1
0 8 18 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation10
0
1
0 8 0 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation14
0
1
0 8 1 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation3
0
1
0 8 2 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation7
0
1
0 8 3 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation8
0
1
0 8 4 12
1
end_operator
begin_operator
turn_to satellite1 star13 phenomenon15
0
1
0 8 5 12
1
end_operator
begin_operator
turn_to satellite1 star13 phenomenon17
0
1
0 8 6 12
1
end_operator
begin_operator
turn_to satellite1 star13 planet16
0
1
0 8 7 12
1
end_operator
begin_operator
turn_to satellite1 star13 star0
0
1
0 8 8 12
1
end_operator
begin_operator
turn_to satellite1 star13 star1
0
1
0 8 9 12
1
end_operator
begin_operator
turn_to satellite1 star13 star11
0
1
0 8 10 12
1
end_operator
begin_operator
turn_to satellite1 star13 star12
0
1
0 8 11 12
1
end_operator
begin_operator
turn_to satellite1 star13 star18
0
1
0 8 13 12
1
end_operator
begin_operator
turn_to satellite1 star13 star2
0
1
0 8 14 12
1
end_operator
begin_operator
turn_to satellite1 star13 star4
0
1
0 8 15 12
1
end_operator
begin_operator
turn_to satellite1 star13 star5
0
1
0 8 16 12
1
end_operator
begin_operator
turn_to satellite1 star13 star6
0
1
0 8 17 12
1
end_operator
begin_operator
turn_to satellite1 star13 star9
0
1
0 8 18 12
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation10
0
1
0 8 0 13
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation14
0
1
0 8 1 13
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation3
0
1
0 8 2 13
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation7
0
1
0 8 3 13
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation8
0
1
0 8 4 13
1
end_operator
begin_operator
turn_to satellite1 star18 phenomenon15
0
1
0 8 5 13
1
end_operator
begin_operator
turn_to satellite1 star18 phenomenon17
0
1
0 8 6 13
1
end_operator
begin_operator
turn_to satellite1 star18 planet16
0
1
0 8 7 13
1
end_operator
begin_operator
turn_to satellite1 star18 star0
0
1
0 8 8 13
1
end_operator
begin_operator
turn_to satellite1 star18 star1
0
1
0 8 9 13
1
end_operator
begin_operator
turn_to satellite1 star18 star11
0
1
0 8 10 13
1
end_operator
begin_operator
turn_to satellite1 star18 star12
0
1
0 8 11 13
1
end_operator
begin_operator
turn_to satellite1 star18 star13
0
1
0 8 12 13
1
end_operator
begin_operator
turn_to satellite1 star18 star2
0
1
0 8 14 13
1
end_operator
begin_operator
turn_to satellite1 star18 star4
0
1
0 8 15 13
1
end_operator
begin_operator
turn_to satellite1 star18 star5
0
1
0 8 16 13
1
end_operator
begin_operator
turn_to satellite1 star18 star6
0
1
0 8 17 13
1
end_operator
begin_operator
turn_to satellite1 star18 star9
0
1
0 8 18 13
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation10
0
1
0 8 0 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation14
0
1
0 8 1 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation3
0
1
0 8 2 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation7
0
1
0 8 3 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation8
0
1
0 8 4 14
1
end_operator
begin_operator
turn_to satellite1 star2 phenomenon15
0
1
0 8 5 14
1
end_operator
begin_operator
turn_to satellite1 star2 phenomenon17
0
1
0 8 6 14
1
end_operator
begin_operator
turn_to satellite1 star2 planet16
0
1
0 8 7 14
1
end_operator
begin_operator
turn_to satellite1 star2 star0
0
1
0 8 8 14
1
end_operator
begin_operator
turn_to satellite1 star2 star1
0
1
0 8 9 14
1
end_operator
begin_operator
turn_to satellite1 star2 star11
0
1
0 8 10 14
1
end_operator
begin_operator
turn_to satellite1 star2 star12
0
1
0 8 11 14
1
end_operator
begin_operator
turn_to satellite1 star2 star13
0
1
0 8 12 14
1
end_operator
begin_operator
turn_to satellite1 star2 star18
0
1
0 8 13 14
1
end_operator
begin_operator
turn_to satellite1 star2 star4
0
1
0 8 15 14
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 8 16 14
1
end_operator
begin_operator
turn_to satellite1 star2 star6
0
1
0 8 17 14
1
end_operator
begin_operator
turn_to satellite1 star2 star9
0
1
0 8 18 14
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation10
0
1
0 8 0 15
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation14
0
1
0 8 1 15
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation3
0
1
0 8 2 15
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation7
0
1
0 8 3 15
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation8
0
1
0 8 4 15
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon15
0
1
0 8 5 15
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon17
0
1
0 8 6 15
1
end_operator
begin_operator
turn_to satellite1 star4 planet16
0
1
0 8 7 15
1
end_operator
begin_operator
turn_to satellite1 star4 star0
0
1
0 8 8 15
1
end_operator
begin_operator
turn_to satellite1 star4 star1
0
1
0 8 9 15
1
end_operator
begin_operator
turn_to satellite1 star4 star11
0
1
0 8 10 15
1
end_operator
begin_operator
turn_to satellite1 star4 star12
0
1
0 8 11 15
1
end_operator
begin_operator
turn_to satellite1 star4 star13
0
1
0 8 12 15
1
end_operator
begin_operator
turn_to satellite1 star4 star18
0
1
0 8 13 15
1
end_operator
begin_operator
turn_to satellite1 star4 star2
0
1
0 8 14 15
1
end_operator
begin_operator
turn_to satellite1 star4 star5
0
1
0 8 16 15
1
end_operator
begin_operator
turn_to satellite1 star4 star6
0
1
0 8 17 15
1
end_operator
begin_operator
turn_to satellite1 star4 star9
0
1
0 8 18 15
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation10
0
1
0 8 0 16
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation14
0
1
0 8 1 16
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation3
0
1
0 8 2 16
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation7
0
1
0 8 3 16
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation8
0
1
0 8 4 16
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon15
0
1
0 8 5 16
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon17
0
1
0 8 6 16
1
end_operator
begin_operator
turn_to satellite1 star5 planet16
0
1
0 8 7 16
1
end_operator
begin_operator
turn_to satellite1 star5 star0
0
1
0 8 8 16
1
end_operator
begin_operator
turn_to satellite1 star5 star1
0
1
0 8 9 16
1
end_operator
begin_operator
turn_to satellite1 star5 star11
0
1
0 8 10 16
1
end_operator
begin_operator
turn_to satellite1 star5 star12
0
1
0 8 11 16
1
end_operator
begin_operator
turn_to satellite1 star5 star13
0
1
0 8 12 16
1
end_operator
begin_operator
turn_to satellite1 star5 star18
0
1
0 8 13 16
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 8 14 16
1
end_operator
begin_operator
turn_to satellite1 star5 star4
0
1
0 8 15 16
1
end_operator
begin_operator
turn_to satellite1 star5 star6
0
1
0 8 17 16
1
end_operator
begin_operator
turn_to satellite1 star5 star9
0
1
0 8 18 16
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation10
0
1
0 8 0 17
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation14
0
1
0 8 1 17
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation3
0
1
0 8 2 17
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation7
0
1
0 8 3 17
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation8
0
1
0 8 4 17
1
end_operator
begin_operator
turn_to satellite1 star6 phenomenon15
0
1
0 8 5 17
1
end_operator
begin_operator
turn_to satellite1 star6 phenomenon17
0
1
0 8 6 17
1
end_operator
begin_operator
turn_to satellite1 star6 planet16
0
1
0 8 7 17
1
end_operator
begin_operator
turn_to satellite1 star6 star0
0
1
0 8 8 17
1
end_operator
begin_operator
turn_to satellite1 star6 star1
0
1
0 8 9 17
1
end_operator
begin_operator
turn_to satellite1 star6 star11
0
1
0 8 10 17
1
end_operator
begin_operator
turn_to satellite1 star6 star12
0
1
0 8 11 17
1
end_operator
begin_operator
turn_to satellite1 star6 star13
0
1
0 8 12 17
1
end_operator
begin_operator
turn_to satellite1 star6 star18
0
1
0 8 13 17
1
end_operator
begin_operator
turn_to satellite1 star6 star2
0
1
0 8 14 17
1
end_operator
begin_operator
turn_to satellite1 star6 star4
0
1
0 8 15 17
1
end_operator
begin_operator
turn_to satellite1 star6 star5
0
1
0 8 16 17
1
end_operator
begin_operator
turn_to satellite1 star6 star9
0
1
0 8 18 17
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation10
0
1
0 8 0 18
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation14
0
1
0 8 1 18
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation3
0
1
0 8 2 18
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation7
0
1
0 8 3 18
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation8
0
1
0 8 4 18
1
end_operator
begin_operator
turn_to satellite1 star9 phenomenon15
0
1
0 8 5 18
1
end_operator
begin_operator
turn_to satellite1 star9 phenomenon17
0
1
0 8 6 18
1
end_operator
begin_operator
turn_to satellite1 star9 planet16
0
1
0 8 7 18
1
end_operator
begin_operator
turn_to satellite1 star9 star0
0
1
0 8 8 18
1
end_operator
begin_operator
turn_to satellite1 star9 star1
0
1
0 8 9 18
1
end_operator
begin_operator
turn_to satellite1 star9 star11
0
1
0 8 10 18
1
end_operator
begin_operator
turn_to satellite1 star9 star12
0
1
0 8 11 18
1
end_operator
begin_operator
turn_to satellite1 star9 star13
0
1
0 8 12 18
1
end_operator
begin_operator
turn_to satellite1 star9 star18
0
1
0 8 13 18
1
end_operator
begin_operator
turn_to satellite1 star9 star2
0
1
0 8 14 18
1
end_operator
begin_operator
turn_to satellite1 star9 star4
0
1
0 8 15 18
1
end_operator
begin_operator
turn_to satellite1 star9 star5
0
1
0 8 16 18
1
end_operator
begin_operator
turn_to satellite1 star9 star6
0
1
0 8 17 18
1
end_operator
begin_operator
turn_to satellite2 groundstation10 groundstation14
0
1
0 7 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 groundstation3
0
1
0 7 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 groundstation7
0
1
0 7 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 groundstation8
0
1
0 7 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 phenomenon15
0
1
0 7 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 phenomenon17
0
1
0 7 6 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 planet16
0
1
0 7 7 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star0
0
1
0 7 8 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star1
0
1
0 7 9 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star11
0
1
0 7 10 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star12
0
1
0 7 11 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star13
0
1
0 7 12 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star18
0
1
0 7 13 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star2
0
1
0 7 14 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star4
0
1
0 7 15 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star5
0
1
0 7 16 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star6
0
1
0 7 17 0
1
end_operator
begin_operator
turn_to satellite2 groundstation10 star9
0
1
0 7 18 0
1
end_operator
begin_operator
turn_to satellite2 groundstation14 groundstation10
0
1
0 7 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 groundstation3
0
1
0 7 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 groundstation7
0
1
0 7 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 groundstation8
0
1
0 7 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 phenomenon15
0
1
0 7 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 phenomenon17
0
1
0 7 6 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 planet16
0
1
0 7 7 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star0
0
1
0 7 8 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star1
0
1
0 7 9 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star11
0
1
0 7 10 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star12
0
1
0 7 11 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star13
0
1
0 7 12 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star18
0
1
0 7 13 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star2
0
1
0 7 14 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star4
0
1
0 7 15 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star5
0
1
0 7 16 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star6
0
1
0 7 17 1
1
end_operator
begin_operator
turn_to satellite2 groundstation14 star9
0
1
0 7 18 1
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation10
0
1
0 7 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation14
0
1
0 7 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation7
0
1
0 7 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation8
0
1
0 7 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 phenomenon15
0
1
0 7 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 phenomenon17
0
1
0 7 6 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 planet16
0
1
0 7 7 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star0
0
1
0 7 8 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star1
0
1
0 7 9 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star11
0
1
0 7 10 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star12
0
1
0 7 11 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star13
0
1
0 7 12 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star18
0
1
0 7 13 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star2
0
1
0 7 14 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star4
0
1
0 7 15 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star5
0
1
0 7 16 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star6
0
1
0 7 17 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star9
0
1
0 7 18 2
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation10
0
1
0 7 0 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation14
0
1
0 7 1 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation3
0
1
0 7 2 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation8
0
1
0 7 4 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 phenomenon15
0
1
0 7 5 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 phenomenon17
0
1
0 7 6 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 planet16
0
1
0 7 7 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star0
0
1
0 7 8 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star1
0
1
0 7 9 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star11
0
1
0 7 10 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star12
0
1
0 7 11 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star13
0
1
0 7 12 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star18
0
1
0 7 13 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star2
0
1
0 7 14 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star4
0
1
0 7 15 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star5
0
1
0 7 16 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star6
0
1
0 7 17 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star9
0
1
0 7 18 3
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation10
0
1
0 7 0 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation14
0
1
0 7 1 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation3
0
1
0 7 2 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation7
0
1
0 7 3 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 phenomenon15
0
1
0 7 5 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 phenomenon17
0
1
0 7 6 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 planet16
0
1
0 7 7 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star0
0
1
0 7 8 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star1
0
1
0 7 9 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star11
0
1
0 7 10 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star12
0
1
0 7 11 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star13
0
1
0 7 12 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star18
0
1
0 7 13 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star2
0
1
0 7 14 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star4
0
1
0 7 15 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star5
0
1
0 7 16 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star6
0
1
0 7 17 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star9
0
1
0 7 18 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 groundstation10
0
1
0 7 0 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 groundstation14
0
1
0 7 1 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 groundstation3
0
1
0 7 2 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 groundstation7
0
1
0 7 3 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 groundstation8
0
1
0 7 4 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 phenomenon17
0
1
0 7 6 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 planet16
0
1
0 7 7 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star0
0
1
0 7 8 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star1
0
1
0 7 9 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star11
0
1
0 7 10 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star12
0
1
0 7 11 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star13
0
1
0 7 12 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star18
0
1
0 7 13 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star2
0
1
0 7 14 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star4
0
1
0 7 15 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star5
0
1
0 7 16 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star6
0
1
0 7 17 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon15 star9
0
1
0 7 18 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 groundstation10
0
1
0 7 0 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 groundstation14
0
1
0 7 1 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 groundstation3
0
1
0 7 2 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 groundstation7
0
1
0 7 3 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 groundstation8
0
1
0 7 4 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 phenomenon15
0
1
0 7 5 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 planet16
0
1
0 7 7 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star0
0
1
0 7 8 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star1
0
1
0 7 9 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star11
0
1
0 7 10 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star12
0
1
0 7 11 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star13
0
1
0 7 12 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star18
0
1
0 7 13 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star2
0
1
0 7 14 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star4
0
1
0 7 15 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star5
0
1
0 7 16 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star6
0
1
0 7 17 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon17 star9
0
1
0 7 18 6
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation10
0
1
0 7 0 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation14
0
1
0 7 1 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation3
0
1
0 7 2 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation7
0
1
0 7 3 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation8
0
1
0 7 4 7
1
end_operator
begin_operator
turn_to satellite2 planet16 phenomenon15
0
1
0 7 5 7
1
end_operator
begin_operator
turn_to satellite2 planet16 phenomenon17
0
1
0 7 6 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star0
0
1
0 7 8 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star1
0
1
0 7 9 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star11
0
1
0 7 10 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star12
0
1
0 7 11 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star13
0
1
0 7 12 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star18
0
1
0 7 13 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star2
0
1
0 7 14 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star4
0
1
0 7 15 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star5
0
1
0 7 16 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star6
0
1
0 7 17 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star9
0
1
0 7 18 7
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation10
0
1
0 7 0 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation14
0
1
0 7 1 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation3
0
1
0 7 2 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation7
0
1
0 7 3 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation8
0
1
0 7 4 8
1
end_operator
begin_operator
turn_to satellite2 star0 phenomenon15
0
1
0 7 5 8
1
end_operator
begin_operator
turn_to satellite2 star0 phenomenon17
0
1
0 7 6 8
1
end_operator
begin_operator
turn_to satellite2 star0 planet16
0
1
0 7 7 8
1
end_operator
begin_operator
turn_to satellite2 star0 star1
0
1
0 7 9 8
1
end_operator
begin_operator
turn_to satellite2 star0 star11
0
1
0 7 10 8
1
end_operator
begin_operator
turn_to satellite2 star0 star12
0
1
0 7 11 8
1
end_operator
begin_operator
turn_to satellite2 star0 star13
0
1
0 7 12 8
1
end_operator
begin_operator
turn_to satellite2 star0 star18
0
1
0 7 13 8
1
end_operator
begin_operator
turn_to satellite2 star0 star2
0
1
0 7 14 8
1
end_operator
begin_operator
turn_to satellite2 star0 star4
0
1
0 7 15 8
1
end_operator
begin_operator
turn_to satellite2 star0 star5
0
1
0 7 16 8
1
end_operator
begin_operator
turn_to satellite2 star0 star6
0
1
0 7 17 8
1
end_operator
begin_operator
turn_to satellite2 star0 star9
0
1
0 7 18 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation10
0
1
0 7 0 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation14
0
1
0 7 1 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation3
0
1
0 7 2 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation7
0
1
0 7 3 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation8
0
1
0 7 4 9
1
end_operator
begin_operator
turn_to satellite2 star1 phenomenon15
0
1
0 7 5 9
1
end_operator
begin_operator
turn_to satellite2 star1 phenomenon17
0
1
0 7 6 9
1
end_operator
begin_operator
turn_to satellite2 star1 planet16
0
1
0 7 7 9
1
end_operator
begin_operator
turn_to satellite2 star1 star0
0
1
0 7 8 9
1
end_operator
begin_operator
turn_to satellite2 star1 star11
0
1
0 7 10 9
1
end_operator
begin_operator
turn_to satellite2 star1 star12
0
1
0 7 11 9
1
end_operator
begin_operator
turn_to satellite2 star1 star13
0
1
0 7 12 9
1
end_operator
begin_operator
turn_to satellite2 star1 star18
0
1
0 7 13 9
1
end_operator
begin_operator
turn_to satellite2 star1 star2
0
1
0 7 14 9
1
end_operator
begin_operator
turn_to satellite2 star1 star4
0
1
0 7 15 9
1
end_operator
begin_operator
turn_to satellite2 star1 star5
0
1
0 7 16 9
1
end_operator
begin_operator
turn_to satellite2 star1 star6
0
1
0 7 17 9
1
end_operator
begin_operator
turn_to satellite2 star1 star9
0
1
0 7 18 9
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation10
0
1
0 7 0 10
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation14
0
1
0 7 1 10
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation3
0
1
0 7 2 10
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation7
0
1
0 7 3 10
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation8
0
1
0 7 4 10
1
end_operator
begin_operator
turn_to satellite2 star11 phenomenon15
0
1
0 7 5 10
1
end_operator
begin_operator
turn_to satellite2 star11 phenomenon17
0
1
0 7 6 10
1
end_operator
begin_operator
turn_to satellite2 star11 planet16
0
1
0 7 7 10
1
end_operator
begin_operator
turn_to satellite2 star11 star0
0
1
0 7 8 10
1
end_operator
begin_operator
turn_to satellite2 star11 star1
0
1
0 7 9 10
1
end_operator
begin_operator
turn_to satellite2 star11 star12
0
1
0 7 11 10
1
end_operator
begin_operator
turn_to satellite2 star11 star13
0
1
0 7 12 10
1
end_operator
begin_operator
turn_to satellite2 star11 star18
0
1
0 7 13 10
1
end_operator
begin_operator
turn_to satellite2 star11 star2
0
1
0 7 14 10
1
end_operator
begin_operator
turn_to satellite2 star11 star4
0
1
0 7 15 10
1
end_operator
begin_operator
turn_to satellite2 star11 star5
0
1
0 7 16 10
1
end_operator
begin_operator
turn_to satellite2 star11 star6
0
1
0 7 17 10
1
end_operator
begin_operator
turn_to satellite2 star11 star9
0
1
0 7 18 10
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation10
0
1
0 7 0 11
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation14
0
1
0 7 1 11
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation3
0
1
0 7 2 11
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation7
0
1
0 7 3 11
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation8
0
1
0 7 4 11
1
end_operator
begin_operator
turn_to satellite2 star12 phenomenon15
0
1
0 7 5 11
1
end_operator
begin_operator
turn_to satellite2 star12 phenomenon17
0
1
0 7 6 11
1
end_operator
begin_operator
turn_to satellite2 star12 planet16
0
1
0 7 7 11
1
end_operator
begin_operator
turn_to satellite2 star12 star0
0
1
0 7 8 11
1
end_operator
begin_operator
turn_to satellite2 star12 star1
0
1
0 7 9 11
1
end_operator
begin_operator
turn_to satellite2 star12 star11
0
1
0 7 10 11
1
end_operator
begin_operator
turn_to satellite2 star12 star13
0
1
0 7 12 11
1
end_operator
begin_operator
turn_to satellite2 star12 star18
0
1
0 7 13 11
1
end_operator
begin_operator
turn_to satellite2 star12 star2
0
1
0 7 14 11
1
end_operator
begin_operator
turn_to satellite2 star12 star4
0
1
0 7 15 11
1
end_operator
begin_operator
turn_to satellite2 star12 star5
0
1
0 7 16 11
1
end_operator
begin_operator
turn_to satellite2 star12 star6
0
1
0 7 17 11
1
end_operator
begin_operator
turn_to satellite2 star12 star9
0
1
0 7 18 11
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation10
0
1
0 7 0 12
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation14
0
1
0 7 1 12
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation3
0
1
0 7 2 12
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation7
0
1
0 7 3 12
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation8
0
1
0 7 4 12
1
end_operator
begin_operator
turn_to satellite2 star13 phenomenon15
0
1
0 7 5 12
1
end_operator
begin_operator
turn_to satellite2 star13 phenomenon17
0
1
0 7 6 12
1
end_operator
begin_operator
turn_to satellite2 star13 planet16
0
1
0 7 7 12
1
end_operator
begin_operator
turn_to satellite2 star13 star0
0
1
0 7 8 12
1
end_operator
begin_operator
turn_to satellite2 star13 star1
0
1
0 7 9 12
1
end_operator
begin_operator
turn_to satellite2 star13 star11
0
1
0 7 10 12
1
end_operator
begin_operator
turn_to satellite2 star13 star12
0
1
0 7 11 12
1
end_operator
begin_operator
turn_to satellite2 star13 star18
0
1
0 7 13 12
1
end_operator
begin_operator
turn_to satellite2 star13 star2
0
1
0 7 14 12
1
end_operator
begin_operator
turn_to satellite2 star13 star4
0
1
0 7 15 12
1
end_operator
begin_operator
turn_to satellite2 star13 star5
0
1
0 7 16 12
1
end_operator
begin_operator
turn_to satellite2 star13 star6
0
1
0 7 17 12
1
end_operator
begin_operator
turn_to satellite2 star13 star9
0
1
0 7 18 12
1
end_operator
begin_operator
turn_to satellite2 star18 groundstation10
0
1
0 7 0 13
1
end_operator
begin_operator
turn_to satellite2 star18 groundstation14
0
1
0 7 1 13
1
end_operator
begin_operator
turn_to satellite2 star18 groundstation3
0
1
0 7 2 13
1
end_operator
begin_operator
turn_to satellite2 star18 groundstation7
0
1
0 7 3 13
1
end_operator
begin_operator
turn_to satellite2 star18 groundstation8
0
1
0 7 4 13
1
end_operator
begin_operator
turn_to satellite2 star18 phenomenon15
0
1
0 7 5 13
1
end_operator
begin_operator
turn_to satellite2 star18 phenomenon17
0
1
0 7 6 13
1
end_operator
begin_operator
turn_to satellite2 star18 planet16
0
1
0 7 7 13
1
end_operator
begin_operator
turn_to satellite2 star18 star0
0
1
0 7 8 13
1
end_operator
begin_operator
turn_to satellite2 star18 star1
0
1
0 7 9 13
1
end_operator
begin_operator
turn_to satellite2 star18 star11
0
1
0 7 10 13
1
end_operator
begin_operator
turn_to satellite2 star18 star12
0
1
0 7 11 13
1
end_operator
begin_operator
turn_to satellite2 star18 star13
0
1
0 7 12 13
1
end_operator
begin_operator
turn_to satellite2 star18 star2
0
1
0 7 14 13
1
end_operator
begin_operator
turn_to satellite2 star18 star4
0
1
0 7 15 13
1
end_operator
begin_operator
turn_to satellite2 star18 star5
0
1
0 7 16 13
1
end_operator
begin_operator
turn_to satellite2 star18 star6
0
1
0 7 17 13
1
end_operator
begin_operator
turn_to satellite2 star18 star9
0
1
0 7 18 13
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation10
0
1
0 7 0 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation14
0
1
0 7 1 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation3
0
1
0 7 2 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation7
0
1
0 7 3 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation8
0
1
0 7 4 14
1
end_operator
begin_operator
turn_to satellite2 star2 phenomenon15
0
1
0 7 5 14
1
end_operator
begin_operator
turn_to satellite2 star2 phenomenon17
0
1
0 7 6 14
1
end_operator
begin_operator
turn_to satellite2 star2 planet16
0
1
0 7 7 14
1
end_operator
begin_operator
turn_to satellite2 star2 star0
0
1
0 7 8 14
1
end_operator
begin_operator
turn_to satellite2 star2 star1
0
1
0 7 9 14
1
end_operator
begin_operator
turn_to satellite2 star2 star11
0
1
0 7 10 14
1
end_operator
begin_operator
turn_to satellite2 star2 star12
0
1
0 7 11 14
1
end_operator
begin_operator
turn_to satellite2 star2 star13
0
1
0 7 12 14
1
end_operator
begin_operator
turn_to satellite2 star2 star18
0
1
0 7 13 14
1
end_operator
begin_operator
turn_to satellite2 star2 star4
0
1
0 7 15 14
1
end_operator
begin_operator
turn_to satellite2 star2 star5
0
1
0 7 16 14
1
end_operator
begin_operator
turn_to satellite2 star2 star6
0
1
0 7 17 14
1
end_operator
begin_operator
turn_to satellite2 star2 star9
0
1
0 7 18 14
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation10
0
1
0 7 0 15
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation14
0
1
0 7 1 15
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation3
0
1
0 7 2 15
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation7
0
1
0 7 3 15
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation8
0
1
0 7 4 15
1
end_operator
begin_operator
turn_to satellite2 star4 phenomenon15
0
1
0 7 5 15
1
end_operator
begin_operator
turn_to satellite2 star4 phenomenon17
0
1
0 7 6 15
1
end_operator
begin_operator
turn_to satellite2 star4 planet16
0
1
0 7 7 15
1
end_operator
begin_operator
turn_to satellite2 star4 star0
0
1
0 7 8 15
1
end_operator
begin_operator
turn_to satellite2 star4 star1
0
1
0 7 9 15
1
end_operator
begin_operator
turn_to satellite2 star4 star11
0
1
0 7 10 15
1
end_operator
begin_operator
turn_to satellite2 star4 star12
0
1
0 7 11 15
1
end_operator
begin_operator
turn_to satellite2 star4 star13
0
1
0 7 12 15
1
end_operator
begin_operator
turn_to satellite2 star4 star18
0
1
0 7 13 15
1
end_operator
begin_operator
turn_to satellite2 star4 star2
0
1
0 7 14 15
1
end_operator
begin_operator
turn_to satellite2 star4 star5
0
1
0 7 16 15
1
end_operator
begin_operator
turn_to satellite2 star4 star6
0
1
0 7 17 15
1
end_operator
begin_operator
turn_to satellite2 star4 star9
0
1
0 7 18 15
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation10
0
1
0 7 0 16
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation14
0
1
0 7 1 16
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation3
0
1
0 7 2 16
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation7
0
1
0 7 3 16
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation8
0
1
0 7 4 16
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon15
0
1
0 7 5 16
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon17
0
1
0 7 6 16
1
end_operator
begin_operator
turn_to satellite2 star5 planet16
0
1
0 7 7 16
1
end_operator
begin_operator
turn_to satellite2 star5 star0
0
1
0 7 8 16
1
end_operator
begin_operator
turn_to satellite2 star5 star1
0
1
0 7 9 16
1
end_operator
begin_operator
turn_to satellite2 star5 star11
0
1
0 7 10 16
1
end_operator
begin_operator
turn_to satellite2 star5 star12
0
1
0 7 11 16
1
end_operator
begin_operator
turn_to satellite2 star5 star13
0
1
0 7 12 16
1
end_operator
begin_operator
turn_to satellite2 star5 star18
0
1
0 7 13 16
1
end_operator
begin_operator
turn_to satellite2 star5 star2
0
1
0 7 14 16
1
end_operator
begin_operator
turn_to satellite2 star5 star4
0
1
0 7 15 16
1
end_operator
begin_operator
turn_to satellite2 star5 star6
0
1
0 7 17 16
1
end_operator
begin_operator
turn_to satellite2 star5 star9
0
1
0 7 18 16
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation10
0
1
0 7 0 17
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation14
0
1
0 7 1 17
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation3
0
1
0 7 2 17
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation7
0
1
0 7 3 17
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation8
0
1
0 7 4 17
1
end_operator
begin_operator
turn_to satellite2 star6 phenomenon15
0
1
0 7 5 17
1
end_operator
begin_operator
turn_to satellite2 star6 phenomenon17
0
1
0 7 6 17
1
end_operator
begin_operator
turn_to satellite2 star6 planet16
0
1
0 7 7 17
1
end_operator
begin_operator
turn_to satellite2 star6 star0
0
1
0 7 8 17
1
end_operator
begin_operator
turn_to satellite2 star6 star1
0
1
0 7 9 17
1
end_operator
begin_operator
turn_to satellite2 star6 star11
0
1
0 7 10 17
1
end_operator
begin_operator
turn_to satellite2 star6 star12
0
1
0 7 11 17
1
end_operator
begin_operator
turn_to satellite2 star6 star13
0
1
0 7 12 17
1
end_operator
begin_operator
turn_to satellite2 star6 star18
0
1
0 7 13 17
1
end_operator
begin_operator
turn_to satellite2 star6 star2
0
1
0 7 14 17
1
end_operator
begin_operator
turn_to satellite2 star6 star4
0
1
0 7 15 17
1
end_operator
begin_operator
turn_to satellite2 star6 star5
0
1
0 7 16 17
1
end_operator
begin_operator
turn_to satellite2 star6 star9
0
1
0 7 18 17
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation10
0
1
0 7 0 18
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation14
0
1
0 7 1 18
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation3
0
1
0 7 2 18
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation7
0
1
0 7 3 18
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation8
0
1
0 7 4 18
1
end_operator
begin_operator
turn_to satellite2 star9 phenomenon15
0
1
0 7 5 18
1
end_operator
begin_operator
turn_to satellite2 star9 phenomenon17
0
1
0 7 6 18
1
end_operator
begin_operator
turn_to satellite2 star9 planet16
0
1
0 7 7 18
1
end_operator
begin_operator
turn_to satellite2 star9 star0
0
1
0 7 8 18
1
end_operator
begin_operator
turn_to satellite2 star9 star1
0
1
0 7 9 18
1
end_operator
begin_operator
turn_to satellite2 star9 star11
0
1
0 7 10 18
1
end_operator
begin_operator
turn_to satellite2 star9 star12
0
1
0 7 11 18
1
end_operator
begin_operator
turn_to satellite2 star9 star13
0
1
0 7 12 18
1
end_operator
begin_operator
turn_to satellite2 star9 star18
0
1
0 7 13 18
1
end_operator
begin_operator
turn_to satellite2 star9 star2
0
1
0 7 14 18
1
end_operator
begin_operator
turn_to satellite2 star9 star4
0
1
0 7 15 18
1
end_operator
begin_operator
turn_to satellite2 star9 star5
0
1
0 7 16 18
1
end_operator
begin_operator
turn_to satellite2 star9 star6
0
1
0 7 17 18
1
end_operator
0
