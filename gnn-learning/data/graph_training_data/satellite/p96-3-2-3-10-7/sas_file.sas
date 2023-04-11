begin_version
3
end_version
begin_metric
0
end_metric
21
begin_variable
var0
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
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
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
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
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
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
17
Atom pointing(satellite2, groundstation4)
Atom pointing(satellite2, groundstation5)
Atom pointing(satellite2, groundstation6)
Atom pointing(satellite2, groundstation7)
Atom pointing(satellite2, groundstation8)
Atom pointing(satellite2, phenomenon10)
Atom pointing(satellite2, planet11)
Atom pointing(satellite2, planet16)
Atom pointing(satellite2, star0)
Atom pointing(satellite2, star1)
Atom pointing(satellite2, star12)
Atom pointing(satellite2, star13)
Atom pointing(satellite2, star14)
Atom pointing(satellite2, star15)
Atom pointing(satellite2, star2)
Atom pointing(satellite2, star3)
Atom pointing(satellite2, star9)
end_variable
begin_variable
var8
-1
17
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, groundstation5)
Atom pointing(satellite1, groundstation6)
Atom pointing(satellite1, groundstation7)
Atom pointing(satellite1, groundstation8)
Atom pointing(satellite1, phenomenon10)
Atom pointing(satellite1, planet11)
Atom pointing(satellite1, planet16)
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star12)
Atom pointing(satellite1, star13)
Atom pointing(satellite1, star14)
Atom pointing(satellite1, star15)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star3)
Atom pointing(satellite1, star9)
end_variable
begin_variable
var9
-1
17
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, groundstation5)
Atom pointing(satellite0, groundstation6)
Atom pointing(satellite0, groundstation7)
Atom pointing(satellite0, groundstation8)
Atom pointing(satellite0, phenomenon10)
Atom pointing(satellite0, planet11)
Atom pointing(satellite0, planet16)
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star12)
Atom pointing(satellite0, star13)
Atom pointing(satellite0, star14)
Atom pointing(satellite0, star15)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star3)
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
Atom have_image(planet16, image0)
NegatedAtom have_image(planet16, image0)
end_variable
begin_variable
var14
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var15
-1
2
Atom have_image(star15, spectrograph1)
NegatedAtom have_image(star15, spectrograph1)
end_variable
begin_variable
var16
-1
2
Atom have_image(star14, spectrograph1)
NegatedAtom have_image(star14, spectrograph1)
end_variable
begin_variable
var17
-1
2
Atom have_image(star13, infrared2)
NegatedAtom have_image(star13, infrared2)
end_variable
begin_variable
var18
-1
2
Atom have_image(star12, infrared2)
NegatedAtom have_image(star12, infrared2)
end_variable
begin_variable
var19
-1
2
Atom have_image(planet11, infrared2)
NegatedAtom have_image(planet11, infrared2)
end_variable
begin_variable
var20
-1
2
Atom have_image(phenomenon10, spectrograph1)
NegatedAtom have_image(phenomenon10, spectrograph1)
end_variable
0
begin_state
0
1
1
1
0
0
1
8
0
8
1
1
1
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
7
13 0
15 0
16 0
17 0
18 0
19 0
20 0
end_goal
853
begin_operator
calibrate satellite0 instrument0 groundstation5
2
9 1
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 groundstation7
2
9 3
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 groundstation8
2
9 4
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation6
2
8 2
2 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star3
2
8 15
2 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 groundstation6
2
8 2
3 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 groundstation4
2
7 0
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
calibrate satellite2 instrument3 star2
2
7 14
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
0 4 -1 0
0 2 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 4 -1 0
0 3 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite2
0
2
0 0 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 14 -1 1
0 5 0 1
0 6 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 12 -1 1
0 4 0 1
0 2 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
3
0 11 -1 1
0 4 0 1
0 3 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite2
0
3
0 10 -1 1
0 0 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon10 instrument0 spectrograph1
3
14 0
9 5
6 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite0 planet11 instrument0 infrared2
3
14 0
9 6
6 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite0 star12 instrument0 infrared2
3
14 0
9 10
6 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star13 instrument0 infrared2
3
14 0
9 11
6 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star14 instrument0 spectrograph1
3
14 0
9 12
6 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star15 instrument0 spectrograph1
3
14 0
9 13
6 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon10 instrument2 spectrograph1
3
11 0
8 5
3 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite1 planet11 instrument1 infrared2
3
12 0
8 6
2 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 planet11 instrument2 infrared2
3
11 0
8 6
3 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 planet16 instrument1 image0
3
12 0
8 7
2 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 star12 instrument1 infrared2
3
12 0
8 10
2 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 star12 instrument2 infrared2
3
11 0
8 10
3 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 star13 instrument1 infrared2
3
12 0
8 11
2 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star13 instrument2 infrared2
3
11 0
8 11
3 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star14 instrument2 spectrograph1
3
11 0
8 12
3 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 star15 instrument2 spectrograph1
3
11 0
8 13
3 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon10 instrument3 spectrograph1
3
10 0
7 5
1 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite2 planet16 instrument3 image0
3
10 0
7 7
1 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite2 star14 instrument3 spectrograph1
3
10 0
7 12
1 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite2 star15 instrument3 spectrograph1
3
10 0
7 13
1 0
1
0 15 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation5
0
1
0 9 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation6
0
1
0 9 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation7
0
1
0 9 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation8
0
1
0 9 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 phenomenon10
0
1
0 9 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet11
0
1
0 9 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet16
0
1
0 9 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star0
0
1
0 9 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star1
0
1
0 9 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star12
0
1
0 9 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star13
0
1
0 9 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star14
0
1
0 9 12 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star15
0
1
0 9 13 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star2
0
1
0 9 14 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star3
0
1
0 9 15 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star9
0
1
0 9 16 0
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation4
0
1
0 9 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation6
0
1
0 9 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation7
0
1
0 9 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 groundstation8
0
1
0 9 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 phenomenon10
0
1
0 9 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 planet11
0
1
0 9 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 planet16
0
1
0 9 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star0
0
1
0 9 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star1
0
1
0 9 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star12
0
1
0 9 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star13
0
1
0 9 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star14
0
1
0 9 12 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star15
0
1
0 9 13 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star2
0
1
0 9 14 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star3
0
1
0 9 15 1
1
end_operator
begin_operator
turn_to satellite0 groundstation5 star9
0
1
0 9 16 1
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation4
0
1
0 9 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation5
0
1
0 9 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation7
0
1
0 9 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation8
0
1
0 9 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon10
0
1
0 9 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 planet11
0
1
0 9 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 planet16
0
1
0 9 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star0
0
1
0 9 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star1
0
1
0 9 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star12
0
1
0 9 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star13
0
1
0 9 11 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star14
0
1
0 9 12 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star15
0
1
0 9 13 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star2
0
1
0 9 14 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star3
0
1
0 9 15 2
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star9
0
1
0 9 16 2
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation4
0
1
0 9 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation5
0
1
0 9 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation6
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
turn_to satellite0 groundstation7 phenomenon10
0
1
0 9 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 planet11
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
turn_to satellite0 groundstation7 star12
0
1
0 9 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star13
0
1
0 9 11 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star14
0
1
0 9 12 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star15
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
turn_to satellite0 groundstation7 star3
0
1
0 9 15 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star9
0
1
0 9 16 3
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation4
0
1
0 9 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation5
0
1
0 9 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation6
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
turn_to satellite0 groundstation8 phenomenon10
0
1
0 9 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 planet11
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
turn_to satellite0 groundstation8 star12
0
1
0 9 10 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star13
0
1
0 9 11 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star14
0
1
0 9 12 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star15
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
turn_to satellite0 groundstation8 star3
0
1
0 9 15 4
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star9
0
1
0 9 16 4
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation4
0
1
0 9 0 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation5
0
1
0 9 1 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation6
0
1
0 9 2 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation7
0
1
0 9 3 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 groundstation8
0
1
0 9 4 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 planet11
0
1
0 9 6 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 planet16
0
1
0 9 7 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star0
0
1
0 9 8 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star1
0
1
0 9 9 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star12
0
1
0 9 10 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star13
0
1
0 9 11 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star14
0
1
0 9 12 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star15
0
1
0 9 13 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star2
0
1
0 9 14 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star3
0
1
0 9 15 5
1
end_operator
begin_operator
turn_to satellite0 phenomenon10 star9
0
1
0 9 16 5
1
end_operator
begin_operator
turn_to satellite0 planet11 groundstation4
0
1
0 9 0 6
1
end_operator
begin_operator
turn_to satellite0 planet11 groundstation5
0
1
0 9 1 6
1
end_operator
begin_operator
turn_to satellite0 planet11 groundstation6
0
1
0 9 2 6
1
end_operator
begin_operator
turn_to satellite0 planet11 groundstation7
0
1
0 9 3 6
1
end_operator
begin_operator
turn_to satellite0 planet11 groundstation8
0
1
0 9 4 6
1
end_operator
begin_operator
turn_to satellite0 planet11 phenomenon10
0
1
0 9 5 6
1
end_operator
begin_operator
turn_to satellite0 planet11 planet16
0
1
0 9 7 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star0
0
1
0 9 8 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star1
0
1
0 9 9 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star12
0
1
0 9 10 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star13
0
1
0 9 11 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star14
0
1
0 9 12 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star15
0
1
0 9 13 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star2
0
1
0 9 14 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star3
0
1
0 9 15 6
1
end_operator
begin_operator
turn_to satellite0 planet11 star9
0
1
0 9 16 6
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation4
0
1
0 9 0 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation5
0
1
0 9 1 7
1
end_operator
begin_operator
turn_to satellite0 planet16 groundstation6
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
turn_to satellite0 planet16 phenomenon10
0
1
0 9 5 7
1
end_operator
begin_operator
turn_to satellite0 planet16 planet11
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
turn_to satellite0 planet16 star12
0
1
0 9 10 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star13
0
1
0 9 11 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star14
0
1
0 9 12 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star15
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
turn_to satellite0 planet16 star3
0
1
0 9 15 7
1
end_operator
begin_operator
turn_to satellite0 planet16 star9
0
1
0 9 16 7
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation4
0
1
0 9 0 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation5
0
1
0 9 1 8
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation6
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
turn_to satellite0 star0 phenomenon10
0
1
0 9 5 8
1
end_operator
begin_operator
turn_to satellite0 star0 planet11
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
turn_to satellite0 star0 star12
0
1
0 9 10 8
1
end_operator
begin_operator
turn_to satellite0 star0 star13
0
1
0 9 11 8
1
end_operator
begin_operator
turn_to satellite0 star0 star14
0
1
0 9 12 8
1
end_operator
begin_operator
turn_to satellite0 star0 star15
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
turn_to satellite0 star0 star3
0
1
0 9 15 8
1
end_operator
begin_operator
turn_to satellite0 star0 star9
0
1
0 9 16 8
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation4
0
1
0 9 0 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation5
0
1
0 9 1 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation6
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
turn_to satellite0 star1 phenomenon10
0
1
0 9 5 9
1
end_operator
begin_operator
turn_to satellite0 star1 planet11
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
turn_to satellite0 star1 star12
0
1
0 9 10 9
1
end_operator
begin_operator
turn_to satellite0 star1 star13
0
1
0 9 11 9
1
end_operator
begin_operator
turn_to satellite0 star1 star14
0
1
0 9 12 9
1
end_operator
begin_operator
turn_to satellite0 star1 star15
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
turn_to satellite0 star1 star3
0
1
0 9 15 9
1
end_operator
begin_operator
turn_to satellite0 star1 star9
0
1
0 9 16 9
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation4
0
1
0 9 0 10
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation5
0
1
0 9 1 10
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation6
0
1
0 9 2 10
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation7
0
1
0 9 3 10
1
end_operator
begin_operator
turn_to satellite0 star12 groundstation8
0
1
0 9 4 10
1
end_operator
begin_operator
turn_to satellite0 star12 phenomenon10
0
1
0 9 5 10
1
end_operator
begin_operator
turn_to satellite0 star12 planet11
0
1
0 9 6 10
1
end_operator
begin_operator
turn_to satellite0 star12 planet16
0
1
0 9 7 10
1
end_operator
begin_operator
turn_to satellite0 star12 star0
0
1
0 9 8 10
1
end_operator
begin_operator
turn_to satellite0 star12 star1
0
1
0 9 9 10
1
end_operator
begin_operator
turn_to satellite0 star12 star13
0
1
0 9 11 10
1
end_operator
begin_operator
turn_to satellite0 star12 star14
0
1
0 9 12 10
1
end_operator
begin_operator
turn_to satellite0 star12 star15
0
1
0 9 13 10
1
end_operator
begin_operator
turn_to satellite0 star12 star2
0
1
0 9 14 10
1
end_operator
begin_operator
turn_to satellite0 star12 star3
0
1
0 9 15 10
1
end_operator
begin_operator
turn_to satellite0 star12 star9
0
1
0 9 16 10
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation4
0
1
0 9 0 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation5
0
1
0 9 1 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation6
0
1
0 9 2 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation7
0
1
0 9 3 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation8
0
1
0 9 4 11
1
end_operator
begin_operator
turn_to satellite0 star13 phenomenon10
0
1
0 9 5 11
1
end_operator
begin_operator
turn_to satellite0 star13 planet11
0
1
0 9 6 11
1
end_operator
begin_operator
turn_to satellite0 star13 planet16
0
1
0 9 7 11
1
end_operator
begin_operator
turn_to satellite0 star13 star0
0
1
0 9 8 11
1
end_operator
begin_operator
turn_to satellite0 star13 star1
0
1
0 9 9 11
1
end_operator
begin_operator
turn_to satellite0 star13 star12
0
1
0 9 10 11
1
end_operator
begin_operator
turn_to satellite0 star13 star14
0
1
0 9 12 11
1
end_operator
begin_operator
turn_to satellite0 star13 star15
0
1
0 9 13 11
1
end_operator
begin_operator
turn_to satellite0 star13 star2
0
1
0 9 14 11
1
end_operator
begin_operator
turn_to satellite0 star13 star3
0
1
0 9 15 11
1
end_operator
begin_operator
turn_to satellite0 star13 star9
0
1
0 9 16 11
1
end_operator
begin_operator
turn_to satellite0 star14 groundstation4
0
1
0 9 0 12
1
end_operator
begin_operator
turn_to satellite0 star14 groundstation5
0
1
0 9 1 12
1
end_operator
begin_operator
turn_to satellite0 star14 groundstation6
0
1
0 9 2 12
1
end_operator
begin_operator
turn_to satellite0 star14 groundstation7
0
1
0 9 3 12
1
end_operator
begin_operator
turn_to satellite0 star14 groundstation8
0
1
0 9 4 12
1
end_operator
begin_operator
turn_to satellite0 star14 phenomenon10
0
1
0 9 5 12
1
end_operator
begin_operator
turn_to satellite0 star14 planet11
0
1
0 9 6 12
1
end_operator
begin_operator
turn_to satellite0 star14 planet16
0
1
0 9 7 12
1
end_operator
begin_operator
turn_to satellite0 star14 star0
0
1
0 9 8 12
1
end_operator
begin_operator
turn_to satellite0 star14 star1
0
1
0 9 9 12
1
end_operator
begin_operator
turn_to satellite0 star14 star12
0
1
0 9 10 12
1
end_operator
begin_operator
turn_to satellite0 star14 star13
0
1
0 9 11 12
1
end_operator
begin_operator
turn_to satellite0 star14 star15
0
1
0 9 13 12
1
end_operator
begin_operator
turn_to satellite0 star14 star2
0
1
0 9 14 12
1
end_operator
begin_operator
turn_to satellite0 star14 star3
0
1
0 9 15 12
1
end_operator
begin_operator
turn_to satellite0 star14 star9
0
1
0 9 16 12
1
end_operator
begin_operator
turn_to satellite0 star15 groundstation4
0
1
0 9 0 13
1
end_operator
begin_operator
turn_to satellite0 star15 groundstation5
0
1
0 9 1 13
1
end_operator
begin_operator
turn_to satellite0 star15 groundstation6
0
1
0 9 2 13
1
end_operator
begin_operator
turn_to satellite0 star15 groundstation7
0
1
0 9 3 13
1
end_operator
begin_operator
turn_to satellite0 star15 groundstation8
0
1
0 9 4 13
1
end_operator
begin_operator
turn_to satellite0 star15 phenomenon10
0
1
0 9 5 13
1
end_operator
begin_operator
turn_to satellite0 star15 planet11
0
1
0 9 6 13
1
end_operator
begin_operator
turn_to satellite0 star15 planet16
0
1
0 9 7 13
1
end_operator
begin_operator
turn_to satellite0 star15 star0
0
1
0 9 8 13
1
end_operator
begin_operator
turn_to satellite0 star15 star1
0
1
0 9 9 13
1
end_operator
begin_operator
turn_to satellite0 star15 star12
0
1
0 9 10 13
1
end_operator
begin_operator
turn_to satellite0 star15 star13
0
1
0 9 11 13
1
end_operator
begin_operator
turn_to satellite0 star15 star14
0
1
0 9 12 13
1
end_operator
begin_operator
turn_to satellite0 star15 star2
0
1
0 9 14 13
1
end_operator
begin_operator
turn_to satellite0 star15 star3
0
1
0 9 15 13
1
end_operator
begin_operator
turn_to satellite0 star15 star9
0
1
0 9 16 13
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation4
0
1
0 9 0 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation5
0
1
0 9 1 14
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation6
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
turn_to satellite0 star2 phenomenon10
0
1
0 9 5 14
1
end_operator
begin_operator
turn_to satellite0 star2 planet11
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
turn_to satellite0 star2 star12
0
1
0 9 10 14
1
end_operator
begin_operator
turn_to satellite0 star2 star13
0
1
0 9 11 14
1
end_operator
begin_operator
turn_to satellite0 star2 star14
0
1
0 9 12 14
1
end_operator
begin_operator
turn_to satellite0 star2 star15
0
1
0 9 13 14
1
end_operator
begin_operator
turn_to satellite0 star2 star3
0
1
0 9 15 14
1
end_operator
begin_operator
turn_to satellite0 star2 star9
0
1
0 9 16 14
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation4
0
1
0 9 0 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation5
0
1
0 9 1 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation6
0
1
0 9 2 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation7
0
1
0 9 3 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation8
0
1
0 9 4 15
1
end_operator
begin_operator
turn_to satellite0 star3 phenomenon10
0
1
0 9 5 15
1
end_operator
begin_operator
turn_to satellite0 star3 planet11
0
1
0 9 6 15
1
end_operator
begin_operator
turn_to satellite0 star3 planet16
0
1
0 9 7 15
1
end_operator
begin_operator
turn_to satellite0 star3 star0
0
1
0 9 8 15
1
end_operator
begin_operator
turn_to satellite0 star3 star1
0
1
0 9 9 15
1
end_operator
begin_operator
turn_to satellite0 star3 star12
0
1
0 9 10 15
1
end_operator
begin_operator
turn_to satellite0 star3 star13
0
1
0 9 11 15
1
end_operator
begin_operator
turn_to satellite0 star3 star14
0
1
0 9 12 15
1
end_operator
begin_operator
turn_to satellite0 star3 star15
0
1
0 9 13 15
1
end_operator
begin_operator
turn_to satellite0 star3 star2
0
1
0 9 14 15
1
end_operator
begin_operator
turn_to satellite0 star3 star9
0
1
0 9 16 15
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation4
0
1
0 9 0 16
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation5
0
1
0 9 1 16
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation6
0
1
0 9 2 16
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation7
0
1
0 9 3 16
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation8
0
1
0 9 4 16
1
end_operator
begin_operator
turn_to satellite0 star9 phenomenon10
0
1
0 9 5 16
1
end_operator
begin_operator
turn_to satellite0 star9 planet11
0
1
0 9 6 16
1
end_operator
begin_operator
turn_to satellite0 star9 planet16
0
1
0 9 7 16
1
end_operator
begin_operator
turn_to satellite0 star9 star0
0
1
0 9 8 16
1
end_operator
begin_operator
turn_to satellite0 star9 star1
0
1
0 9 9 16
1
end_operator
begin_operator
turn_to satellite0 star9 star12
0
1
0 9 10 16
1
end_operator
begin_operator
turn_to satellite0 star9 star13
0
1
0 9 11 16
1
end_operator
begin_operator
turn_to satellite0 star9 star14
0
1
0 9 12 16
1
end_operator
begin_operator
turn_to satellite0 star9 star15
0
1
0 9 13 16
1
end_operator
begin_operator
turn_to satellite0 star9 star2
0
1
0 9 14 16
1
end_operator
begin_operator
turn_to satellite0 star9 star3
0
1
0 9 15 16
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation5
0
1
0 8 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation6
0
1
0 8 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation7
0
1
0 8 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation8
0
1
0 8 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 phenomenon10
0
1
0 8 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet11
0
1
0 8 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet16
0
1
0 8 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star0
0
1
0 8 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star1
0
1
0 8 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star12
0
1
0 8 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star13
0
1
0 8 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star14
0
1
0 8 12 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star15
0
1
0 8 13 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star2
0
1
0 8 14 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star3
0
1
0 8 15 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star9
0
1
0 8 16 0
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation4
0
1
0 8 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation6
0
1
0 8 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation7
0
1
0 8 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 groundstation8
0
1
0 8 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 phenomenon10
0
1
0 8 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 planet11
0
1
0 8 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 planet16
0
1
0 8 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star0
0
1
0 8 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star1
0
1
0 8 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star12
0
1
0 8 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star13
0
1
0 8 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star14
0
1
0 8 12 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star15
0
1
0 8 13 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star2
0
1
0 8 14 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star3
0
1
0 8 15 1
1
end_operator
begin_operator
turn_to satellite1 groundstation5 star9
0
1
0 8 16 1
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation4
0
1
0 8 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation5
0
1
0 8 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation7
0
1
0 8 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation8
0
1
0 8 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon10
0
1
0 8 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 planet11
0
1
0 8 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 planet16
0
1
0 8 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star0
0
1
0 8 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star1
0
1
0 8 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star12
0
1
0 8 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star13
0
1
0 8 11 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star14
0
1
0 8 12 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star15
0
1
0 8 13 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star2
0
1
0 8 14 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star3
0
1
0 8 15 2
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star9
0
1
0 8 16 2
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation4
0
1
0 8 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation5
0
1
0 8 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation6
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
turn_to satellite1 groundstation7 phenomenon10
0
1
0 8 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 planet11
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
turn_to satellite1 groundstation7 star12
0
1
0 8 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star13
0
1
0 8 11 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star14
0
1
0 8 12 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star15
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
turn_to satellite1 groundstation7 star3
0
1
0 8 15 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star9
0
1
0 8 16 3
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation4
0
1
0 8 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation5
0
1
0 8 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation6
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
turn_to satellite1 groundstation8 phenomenon10
0
1
0 8 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 planet11
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
turn_to satellite1 groundstation8 star12
0
1
0 8 10 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star13
0
1
0 8 11 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star14
0
1
0 8 12 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star15
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
turn_to satellite1 groundstation8 star3
0
1
0 8 15 4
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star9
0
1
0 8 16 4
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation4
0
1
0 8 0 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation5
0
1
0 8 1 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation6
0
1
0 8 2 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation7
0
1
0 8 3 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 groundstation8
0
1
0 8 4 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 planet11
0
1
0 8 6 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 planet16
0
1
0 8 7 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star0
0
1
0 8 8 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star1
0
1
0 8 9 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star12
0
1
0 8 10 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star13
0
1
0 8 11 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star14
0
1
0 8 12 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star15
0
1
0 8 13 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star2
0
1
0 8 14 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star3
0
1
0 8 15 5
1
end_operator
begin_operator
turn_to satellite1 phenomenon10 star9
0
1
0 8 16 5
1
end_operator
begin_operator
turn_to satellite1 planet11 groundstation4
0
1
0 8 0 6
1
end_operator
begin_operator
turn_to satellite1 planet11 groundstation5
0
1
0 8 1 6
1
end_operator
begin_operator
turn_to satellite1 planet11 groundstation6
0
1
0 8 2 6
1
end_operator
begin_operator
turn_to satellite1 planet11 groundstation7
0
1
0 8 3 6
1
end_operator
begin_operator
turn_to satellite1 planet11 groundstation8
0
1
0 8 4 6
1
end_operator
begin_operator
turn_to satellite1 planet11 phenomenon10
0
1
0 8 5 6
1
end_operator
begin_operator
turn_to satellite1 planet11 planet16
0
1
0 8 7 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star0
0
1
0 8 8 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star1
0
1
0 8 9 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star12
0
1
0 8 10 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star13
0
1
0 8 11 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star14
0
1
0 8 12 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star15
0
1
0 8 13 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star2
0
1
0 8 14 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star3
0
1
0 8 15 6
1
end_operator
begin_operator
turn_to satellite1 planet11 star9
0
1
0 8 16 6
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation4
0
1
0 8 0 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation5
0
1
0 8 1 7
1
end_operator
begin_operator
turn_to satellite1 planet16 groundstation6
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
turn_to satellite1 planet16 phenomenon10
0
1
0 8 5 7
1
end_operator
begin_operator
turn_to satellite1 planet16 planet11
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
turn_to satellite1 planet16 star12
0
1
0 8 10 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star13
0
1
0 8 11 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star14
0
1
0 8 12 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star15
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
turn_to satellite1 planet16 star3
0
1
0 8 15 7
1
end_operator
begin_operator
turn_to satellite1 planet16 star9
0
1
0 8 16 7
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation4
0
1
0 8 0 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation5
0
1
0 8 1 8
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation6
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
turn_to satellite1 star0 phenomenon10
0
1
0 8 5 8
1
end_operator
begin_operator
turn_to satellite1 star0 planet11
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
turn_to satellite1 star0 star12
0
1
0 8 10 8
1
end_operator
begin_operator
turn_to satellite1 star0 star13
0
1
0 8 11 8
1
end_operator
begin_operator
turn_to satellite1 star0 star14
0
1
0 8 12 8
1
end_operator
begin_operator
turn_to satellite1 star0 star15
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
turn_to satellite1 star0 star3
0
1
0 8 15 8
1
end_operator
begin_operator
turn_to satellite1 star0 star9
0
1
0 8 16 8
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation4
0
1
0 8 0 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation5
0
1
0 8 1 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation6
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
turn_to satellite1 star1 phenomenon10
0
1
0 8 5 9
1
end_operator
begin_operator
turn_to satellite1 star1 planet11
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
turn_to satellite1 star1 star12
0
1
0 8 10 9
1
end_operator
begin_operator
turn_to satellite1 star1 star13
0
1
0 8 11 9
1
end_operator
begin_operator
turn_to satellite1 star1 star14
0
1
0 8 12 9
1
end_operator
begin_operator
turn_to satellite1 star1 star15
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
turn_to satellite1 star1 star3
0
1
0 8 15 9
1
end_operator
begin_operator
turn_to satellite1 star1 star9
0
1
0 8 16 9
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation4
0
1
0 8 0 10
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation5
0
1
0 8 1 10
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation6
0
1
0 8 2 10
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation7
0
1
0 8 3 10
1
end_operator
begin_operator
turn_to satellite1 star12 groundstation8
0
1
0 8 4 10
1
end_operator
begin_operator
turn_to satellite1 star12 phenomenon10
0
1
0 8 5 10
1
end_operator
begin_operator
turn_to satellite1 star12 planet11
0
1
0 8 6 10
1
end_operator
begin_operator
turn_to satellite1 star12 planet16
0
1
0 8 7 10
1
end_operator
begin_operator
turn_to satellite1 star12 star0
0
1
0 8 8 10
1
end_operator
begin_operator
turn_to satellite1 star12 star1
0
1
0 8 9 10
1
end_operator
begin_operator
turn_to satellite1 star12 star13
0
1
0 8 11 10
1
end_operator
begin_operator
turn_to satellite1 star12 star14
0
1
0 8 12 10
1
end_operator
begin_operator
turn_to satellite1 star12 star15
0
1
0 8 13 10
1
end_operator
begin_operator
turn_to satellite1 star12 star2
0
1
0 8 14 10
1
end_operator
begin_operator
turn_to satellite1 star12 star3
0
1
0 8 15 10
1
end_operator
begin_operator
turn_to satellite1 star12 star9
0
1
0 8 16 10
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation4
0
1
0 8 0 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation5
0
1
0 8 1 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation6
0
1
0 8 2 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation7
0
1
0 8 3 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation8
0
1
0 8 4 11
1
end_operator
begin_operator
turn_to satellite1 star13 phenomenon10
0
1
0 8 5 11
1
end_operator
begin_operator
turn_to satellite1 star13 planet11
0
1
0 8 6 11
1
end_operator
begin_operator
turn_to satellite1 star13 planet16
0
1
0 8 7 11
1
end_operator
begin_operator
turn_to satellite1 star13 star0
0
1
0 8 8 11
1
end_operator
begin_operator
turn_to satellite1 star13 star1
0
1
0 8 9 11
1
end_operator
begin_operator
turn_to satellite1 star13 star12
0
1
0 8 10 11
1
end_operator
begin_operator
turn_to satellite1 star13 star14
0
1
0 8 12 11
1
end_operator
begin_operator
turn_to satellite1 star13 star15
0
1
0 8 13 11
1
end_operator
begin_operator
turn_to satellite1 star13 star2
0
1
0 8 14 11
1
end_operator
begin_operator
turn_to satellite1 star13 star3
0
1
0 8 15 11
1
end_operator
begin_operator
turn_to satellite1 star13 star9
0
1
0 8 16 11
1
end_operator
begin_operator
turn_to satellite1 star14 groundstation4
0
1
0 8 0 12
1
end_operator
begin_operator
turn_to satellite1 star14 groundstation5
0
1
0 8 1 12
1
end_operator
begin_operator
turn_to satellite1 star14 groundstation6
0
1
0 8 2 12
1
end_operator
begin_operator
turn_to satellite1 star14 groundstation7
0
1
0 8 3 12
1
end_operator
begin_operator
turn_to satellite1 star14 groundstation8
0
1
0 8 4 12
1
end_operator
begin_operator
turn_to satellite1 star14 phenomenon10
0
1
0 8 5 12
1
end_operator
begin_operator
turn_to satellite1 star14 planet11
0
1
0 8 6 12
1
end_operator
begin_operator
turn_to satellite1 star14 planet16
0
1
0 8 7 12
1
end_operator
begin_operator
turn_to satellite1 star14 star0
0
1
0 8 8 12
1
end_operator
begin_operator
turn_to satellite1 star14 star1
0
1
0 8 9 12
1
end_operator
begin_operator
turn_to satellite1 star14 star12
0
1
0 8 10 12
1
end_operator
begin_operator
turn_to satellite1 star14 star13
0
1
0 8 11 12
1
end_operator
begin_operator
turn_to satellite1 star14 star15
0
1
0 8 13 12
1
end_operator
begin_operator
turn_to satellite1 star14 star2
0
1
0 8 14 12
1
end_operator
begin_operator
turn_to satellite1 star14 star3
0
1
0 8 15 12
1
end_operator
begin_operator
turn_to satellite1 star14 star9
0
1
0 8 16 12
1
end_operator
begin_operator
turn_to satellite1 star15 groundstation4
0
1
0 8 0 13
1
end_operator
begin_operator
turn_to satellite1 star15 groundstation5
0
1
0 8 1 13
1
end_operator
begin_operator
turn_to satellite1 star15 groundstation6
0
1
0 8 2 13
1
end_operator
begin_operator
turn_to satellite1 star15 groundstation7
0
1
0 8 3 13
1
end_operator
begin_operator
turn_to satellite1 star15 groundstation8
0
1
0 8 4 13
1
end_operator
begin_operator
turn_to satellite1 star15 phenomenon10
0
1
0 8 5 13
1
end_operator
begin_operator
turn_to satellite1 star15 planet11
0
1
0 8 6 13
1
end_operator
begin_operator
turn_to satellite1 star15 planet16
0
1
0 8 7 13
1
end_operator
begin_operator
turn_to satellite1 star15 star0
0
1
0 8 8 13
1
end_operator
begin_operator
turn_to satellite1 star15 star1
0
1
0 8 9 13
1
end_operator
begin_operator
turn_to satellite1 star15 star12
0
1
0 8 10 13
1
end_operator
begin_operator
turn_to satellite1 star15 star13
0
1
0 8 11 13
1
end_operator
begin_operator
turn_to satellite1 star15 star14
0
1
0 8 12 13
1
end_operator
begin_operator
turn_to satellite1 star15 star2
0
1
0 8 14 13
1
end_operator
begin_operator
turn_to satellite1 star15 star3
0
1
0 8 15 13
1
end_operator
begin_operator
turn_to satellite1 star15 star9
0
1
0 8 16 13
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation4
0
1
0 8 0 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation5
0
1
0 8 1 14
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation6
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
turn_to satellite1 star2 phenomenon10
0
1
0 8 5 14
1
end_operator
begin_operator
turn_to satellite1 star2 planet11
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
turn_to satellite1 star2 star12
0
1
0 8 10 14
1
end_operator
begin_operator
turn_to satellite1 star2 star13
0
1
0 8 11 14
1
end_operator
begin_operator
turn_to satellite1 star2 star14
0
1
0 8 12 14
1
end_operator
begin_operator
turn_to satellite1 star2 star15
0
1
0 8 13 14
1
end_operator
begin_operator
turn_to satellite1 star2 star3
0
1
0 8 15 14
1
end_operator
begin_operator
turn_to satellite1 star2 star9
0
1
0 8 16 14
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation4
0
1
0 8 0 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation5
0
1
0 8 1 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation6
0
1
0 8 2 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation7
0
1
0 8 3 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation8
0
1
0 8 4 15
1
end_operator
begin_operator
turn_to satellite1 star3 phenomenon10
0
1
0 8 5 15
1
end_operator
begin_operator
turn_to satellite1 star3 planet11
0
1
0 8 6 15
1
end_operator
begin_operator
turn_to satellite1 star3 planet16
0
1
0 8 7 15
1
end_operator
begin_operator
turn_to satellite1 star3 star0
0
1
0 8 8 15
1
end_operator
begin_operator
turn_to satellite1 star3 star1
0
1
0 8 9 15
1
end_operator
begin_operator
turn_to satellite1 star3 star12
0
1
0 8 10 15
1
end_operator
begin_operator
turn_to satellite1 star3 star13
0
1
0 8 11 15
1
end_operator
begin_operator
turn_to satellite1 star3 star14
0
1
0 8 12 15
1
end_operator
begin_operator
turn_to satellite1 star3 star15
0
1
0 8 13 15
1
end_operator
begin_operator
turn_to satellite1 star3 star2
0
1
0 8 14 15
1
end_operator
begin_operator
turn_to satellite1 star3 star9
0
1
0 8 16 15
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation4
0
1
0 8 0 16
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation5
0
1
0 8 1 16
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation6
0
1
0 8 2 16
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation7
0
1
0 8 3 16
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation8
0
1
0 8 4 16
1
end_operator
begin_operator
turn_to satellite1 star9 phenomenon10
0
1
0 8 5 16
1
end_operator
begin_operator
turn_to satellite1 star9 planet11
0
1
0 8 6 16
1
end_operator
begin_operator
turn_to satellite1 star9 planet16
0
1
0 8 7 16
1
end_operator
begin_operator
turn_to satellite1 star9 star0
0
1
0 8 8 16
1
end_operator
begin_operator
turn_to satellite1 star9 star1
0
1
0 8 9 16
1
end_operator
begin_operator
turn_to satellite1 star9 star12
0
1
0 8 10 16
1
end_operator
begin_operator
turn_to satellite1 star9 star13
0
1
0 8 11 16
1
end_operator
begin_operator
turn_to satellite1 star9 star14
0
1
0 8 12 16
1
end_operator
begin_operator
turn_to satellite1 star9 star15
0
1
0 8 13 16
1
end_operator
begin_operator
turn_to satellite1 star9 star2
0
1
0 8 14 16
1
end_operator
begin_operator
turn_to satellite1 star9 star3
0
1
0 8 15 16
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation5
0
1
0 7 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation6
0
1
0 7 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation7
0
1
0 7 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation8
0
1
0 7 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 phenomenon10
0
1
0 7 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 planet11
0
1
0 7 6 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 planet16
0
1
0 7 7 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star0
0
1
0 7 8 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star1
0
1
0 7 9 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star12
0
1
0 7 10 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star13
0
1
0 7 11 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star14
0
1
0 7 12 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star15
0
1
0 7 13 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star2
0
1
0 7 14 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star3
0
1
0 7 15 0
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star9
0
1
0 7 16 0
1
end_operator
begin_operator
turn_to satellite2 groundstation5 groundstation4
0
1
0 7 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 groundstation6
0
1
0 7 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 groundstation7
0
1
0 7 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 groundstation8
0
1
0 7 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 phenomenon10
0
1
0 7 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 planet11
0
1
0 7 6 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 planet16
0
1
0 7 7 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star0
0
1
0 7 8 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star1
0
1
0 7 9 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star12
0
1
0 7 10 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star13
0
1
0 7 11 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star14
0
1
0 7 12 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star15
0
1
0 7 13 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star2
0
1
0 7 14 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star3
0
1
0 7 15 1
1
end_operator
begin_operator
turn_to satellite2 groundstation5 star9
0
1
0 7 16 1
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation4
0
1
0 7 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation5
0
1
0 7 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation7
0
1
0 7 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation8
0
1
0 7 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 phenomenon10
0
1
0 7 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 planet11
0
1
0 7 6 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 planet16
0
1
0 7 7 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star0
0
1
0 7 8 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star1
0
1
0 7 9 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star12
0
1
0 7 10 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star13
0
1
0 7 11 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star14
0
1
0 7 12 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star15
0
1
0 7 13 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star2
0
1
0 7 14 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star3
0
1
0 7 15 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star9
0
1
0 7 16 2
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation4
0
1
0 7 0 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation5
0
1
0 7 1 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation6
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
turn_to satellite2 groundstation7 phenomenon10
0
1
0 7 5 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 planet11
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
turn_to satellite2 groundstation7 star12
0
1
0 7 10 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star13
0
1
0 7 11 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star14
0
1
0 7 12 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star15
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
turn_to satellite2 groundstation7 star3
0
1
0 7 15 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star9
0
1
0 7 16 3
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation4
0
1
0 7 0 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation5
0
1
0 7 1 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 groundstation6
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
turn_to satellite2 groundstation8 phenomenon10
0
1
0 7 5 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 planet11
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
turn_to satellite2 groundstation8 star12
0
1
0 7 10 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star13
0
1
0 7 11 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star14
0
1
0 7 12 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star15
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
turn_to satellite2 groundstation8 star3
0
1
0 7 15 4
1
end_operator
begin_operator
turn_to satellite2 groundstation8 star9
0
1
0 7 16 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 groundstation4
0
1
0 7 0 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 groundstation5
0
1
0 7 1 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 groundstation6
0
1
0 7 2 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 groundstation7
0
1
0 7 3 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 groundstation8
0
1
0 7 4 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 planet11
0
1
0 7 6 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 planet16
0
1
0 7 7 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star0
0
1
0 7 8 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star1
0
1
0 7 9 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star12
0
1
0 7 10 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star13
0
1
0 7 11 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star14
0
1
0 7 12 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star15
0
1
0 7 13 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star2
0
1
0 7 14 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star3
0
1
0 7 15 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon10 star9
0
1
0 7 16 5
1
end_operator
begin_operator
turn_to satellite2 planet11 groundstation4
0
1
0 7 0 6
1
end_operator
begin_operator
turn_to satellite2 planet11 groundstation5
0
1
0 7 1 6
1
end_operator
begin_operator
turn_to satellite2 planet11 groundstation6
0
1
0 7 2 6
1
end_operator
begin_operator
turn_to satellite2 planet11 groundstation7
0
1
0 7 3 6
1
end_operator
begin_operator
turn_to satellite2 planet11 groundstation8
0
1
0 7 4 6
1
end_operator
begin_operator
turn_to satellite2 planet11 phenomenon10
0
1
0 7 5 6
1
end_operator
begin_operator
turn_to satellite2 planet11 planet16
0
1
0 7 7 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star0
0
1
0 7 8 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star1
0
1
0 7 9 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star12
0
1
0 7 10 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star13
0
1
0 7 11 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star14
0
1
0 7 12 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star15
0
1
0 7 13 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star2
0
1
0 7 14 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star3
0
1
0 7 15 6
1
end_operator
begin_operator
turn_to satellite2 planet11 star9
0
1
0 7 16 6
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation4
0
1
0 7 0 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation5
0
1
0 7 1 7
1
end_operator
begin_operator
turn_to satellite2 planet16 groundstation6
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
turn_to satellite2 planet16 phenomenon10
0
1
0 7 5 7
1
end_operator
begin_operator
turn_to satellite2 planet16 planet11
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
turn_to satellite2 planet16 star12
0
1
0 7 10 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star13
0
1
0 7 11 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star14
0
1
0 7 12 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star15
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
turn_to satellite2 planet16 star3
0
1
0 7 15 7
1
end_operator
begin_operator
turn_to satellite2 planet16 star9
0
1
0 7 16 7
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation4
0
1
0 7 0 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation5
0
1
0 7 1 8
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation6
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
turn_to satellite2 star0 phenomenon10
0
1
0 7 5 8
1
end_operator
begin_operator
turn_to satellite2 star0 planet11
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
turn_to satellite2 star0 star12
0
1
0 7 10 8
1
end_operator
begin_operator
turn_to satellite2 star0 star13
0
1
0 7 11 8
1
end_operator
begin_operator
turn_to satellite2 star0 star14
0
1
0 7 12 8
1
end_operator
begin_operator
turn_to satellite2 star0 star15
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
turn_to satellite2 star0 star3
0
1
0 7 15 8
1
end_operator
begin_operator
turn_to satellite2 star0 star9
0
1
0 7 16 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation4
0
1
0 7 0 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation5
0
1
0 7 1 9
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation6
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
turn_to satellite2 star1 phenomenon10
0
1
0 7 5 9
1
end_operator
begin_operator
turn_to satellite2 star1 planet11
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
turn_to satellite2 star1 star12
0
1
0 7 10 9
1
end_operator
begin_operator
turn_to satellite2 star1 star13
0
1
0 7 11 9
1
end_operator
begin_operator
turn_to satellite2 star1 star14
0
1
0 7 12 9
1
end_operator
begin_operator
turn_to satellite2 star1 star15
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
turn_to satellite2 star1 star3
0
1
0 7 15 9
1
end_operator
begin_operator
turn_to satellite2 star1 star9
0
1
0 7 16 9
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation4
0
1
0 7 0 10
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation5
0
1
0 7 1 10
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation6
0
1
0 7 2 10
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation7
0
1
0 7 3 10
1
end_operator
begin_operator
turn_to satellite2 star12 groundstation8
0
1
0 7 4 10
1
end_operator
begin_operator
turn_to satellite2 star12 phenomenon10
0
1
0 7 5 10
1
end_operator
begin_operator
turn_to satellite2 star12 planet11
0
1
0 7 6 10
1
end_operator
begin_operator
turn_to satellite2 star12 planet16
0
1
0 7 7 10
1
end_operator
begin_operator
turn_to satellite2 star12 star0
0
1
0 7 8 10
1
end_operator
begin_operator
turn_to satellite2 star12 star1
0
1
0 7 9 10
1
end_operator
begin_operator
turn_to satellite2 star12 star13
0
1
0 7 11 10
1
end_operator
begin_operator
turn_to satellite2 star12 star14
0
1
0 7 12 10
1
end_operator
begin_operator
turn_to satellite2 star12 star15
0
1
0 7 13 10
1
end_operator
begin_operator
turn_to satellite2 star12 star2
0
1
0 7 14 10
1
end_operator
begin_operator
turn_to satellite2 star12 star3
0
1
0 7 15 10
1
end_operator
begin_operator
turn_to satellite2 star12 star9
0
1
0 7 16 10
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation4
0
1
0 7 0 11
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation5
0
1
0 7 1 11
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation6
0
1
0 7 2 11
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation7
0
1
0 7 3 11
1
end_operator
begin_operator
turn_to satellite2 star13 groundstation8
0
1
0 7 4 11
1
end_operator
begin_operator
turn_to satellite2 star13 phenomenon10
0
1
0 7 5 11
1
end_operator
begin_operator
turn_to satellite2 star13 planet11
0
1
0 7 6 11
1
end_operator
begin_operator
turn_to satellite2 star13 planet16
0
1
0 7 7 11
1
end_operator
begin_operator
turn_to satellite2 star13 star0
0
1
0 7 8 11
1
end_operator
begin_operator
turn_to satellite2 star13 star1
0
1
0 7 9 11
1
end_operator
begin_operator
turn_to satellite2 star13 star12
0
1
0 7 10 11
1
end_operator
begin_operator
turn_to satellite2 star13 star14
0
1
0 7 12 11
1
end_operator
begin_operator
turn_to satellite2 star13 star15
0
1
0 7 13 11
1
end_operator
begin_operator
turn_to satellite2 star13 star2
0
1
0 7 14 11
1
end_operator
begin_operator
turn_to satellite2 star13 star3
0
1
0 7 15 11
1
end_operator
begin_operator
turn_to satellite2 star13 star9
0
1
0 7 16 11
1
end_operator
begin_operator
turn_to satellite2 star14 groundstation4
0
1
0 7 0 12
1
end_operator
begin_operator
turn_to satellite2 star14 groundstation5
0
1
0 7 1 12
1
end_operator
begin_operator
turn_to satellite2 star14 groundstation6
0
1
0 7 2 12
1
end_operator
begin_operator
turn_to satellite2 star14 groundstation7
0
1
0 7 3 12
1
end_operator
begin_operator
turn_to satellite2 star14 groundstation8
0
1
0 7 4 12
1
end_operator
begin_operator
turn_to satellite2 star14 phenomenon10
0
1
0 7 5 12
1
end_operator
begin_operator
turn_to satellite2 star14 planet11
0
1
0 7 6 12
1
end_operator
begin_operator
turn_to satellite2 star14 planet16
0
1
0 7 7 12
1
end_operator
begin_operator
turn_to satellite2 star14 star0
0
1
0 7 8 12
1
end_operator
begin_operator
turn_to satellite2 star14 star1
0
1
0 7 9 12
1
end_operator
begin_operator
turn_to satellite2 star14 star12
0
1
0 7 10 12
1
end_operator
begin_operator
turn_to satellite2 star14 star13
0
1
0 7 11 12
1
end_operator
begin_operator
turn_to satellite2 star14 star15
0
1
0 7 13 12
1
end_operator
begin_operator
turn_to satellite2 star14 star2
0
1
0 7 14 12
1
end_operator
begin_operator
turn_to satellite2 star14 star3
0
1
0 7 15 12
1
end_operator
begin_operator
turn_to satellite2 star14 star9
0
1
0 7 16 12
1
end_operator
begin_operator
turn_to satellite2 star15 groundstation4
0
1
0 7 0 13
1
end_operator
begin_operator
turn_to satellite2 star15 groundstation5
0
1
0 7 1 13
1
end_operator
begin_operator
turn_to satellite2 star15 groundstation6
0
1
0 7 2 13
1
end_operator
begin_operator
turn_to satellite2 star15 groundstation7
0
1
0 7 3 13
1
end_operator
begin_operator
turn_to satellite2 star15 groundstation8
0
1
0 7 4 13
1
end_operator
begin_operator
turn_to satellite2 star15 phenomenon10
0
1
0 7 5 13
1
end_operator
begin_operator
turn_to satellite2 star15 planet11
0
1
0 7 6 13
1
end_operator
begin_operator
turn_to satellite2 star15 planet16
0
1
0 7 7 13
1
end_operator
begin_operator
turn_to satellite2 star15 star0
0
1
0 7 8 13
1
end_operator
begin_operator
turn_to satellite2 star15 star1
0
1
0 7 9 13
1
end_operator
begin_operator
turn_to satellite2 star15 star12
0
1
0 7 10 13
1
end_operator
begin_operator
turn_to satellite2 star15 star13
0
1
0 7 11 13
1
end_operator
begin_operator
turn_to satellite2 star15 star14
0
1
0 7 12 13
1
end_operator
begin_operator
turn_to satellite2 star15 star2
0
1
0 7 14 13
1
end_operator
begin_operator
turn_to satellite2 star15 star3
0
1
0 7 15 13
1
end_operator
begin_operator
turn_to satellite2 star15 star9
0
1
0 7 16 13
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation4
0
1
0 7 0 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation5
0
1
0 7 1 14
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation6
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
turn_to satellite2 star2 phenomenon10
0
1
0 7 5 14
1
end_operator
begin_operator
turn_to satellite2 star2 planet11
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
turn_to satellite2 star2 star12
0
1
0 7 10 14
1
end_operator
begin_operator
turn_to satellite2 star2 star13
0
1
0 7 11 14
1
end_operator
begin_operator
turn_to satellite2 star2 star14
0
1
0 7 12 14
1
end_operator
begin_operator
turn_to satellite2 star2 star15
0
1
0 7 13 14
1
end_operator
begin_operator
turn_to satellite2 star2 star3
0
1
0 7 15 14
1
end_operator
begin_operator
turn_to satellite2 star2 star9
0
1
0 7 16 14
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation4
0
1
0 7 0 15
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation5
0
1
0 7 1 15
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation6
0
1
0 7 2 15
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation7
0
1
0 7 3 15
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation8
0
1
0 7 4 15
1
end_operator
begin_operator
turn_to satellite2 star3 phenomenon10
0
1
0 7 5 15
1
end_operator
begin_operator
turn_to satellite2 star3 planet11
0
1
0 7 6 15
1
end_operator
begin_operator
turn_to satellite2 star3 planet16
0
1
0 7 7 15
1
end_operator
begin_operator
turn_to satellite2 star3 star0
0
1
0 7 8 15
1
end_operator
begin_operator
turn_to satellite2 star3 star1
0
1
0 7 9 15
1
end_operator
begin_operator
turn_to satellite2 star3 star12
0
1
0 7 10 15
1
end_operator
begin_operator
turn_to satellite2 star3 star13
0
1
0 7 11 15
1
end_operator
begin_operator
turn_to satellite2 star3 star14
0
1
0 7 12 15
1
end_operator
begin_operator
turn_to satellite2 star3 star15
0
1
0 7 13 15
1
end_operator
begin_operator
turn_to satellite2 star3 star2
0
1
0 7 14 15
1
end_operator
begin_operator
turn_to satellite2 star3 star9
0
1
0 7 16 15
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation4
0
1
0 7 0 16
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation5
0
1
0 7 1 16
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation6
0
1
0 7 2 16
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation7
0
1
0 7 3 16
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation8
0
1
0 7 4 16
1
end_operator
begin_operator
turn_to satellite2 star9 phenomenon10
0
1
0 7 5 16
1
end_operator
begin_operator
turn_to satellite2 star9 planet11
0
1
0 7 6 16
1
end_operator
begin_operator
turn_to satellite2 star9 planet16
0
1
0 7 7 16
1
end_operator
begin_operator
turn_to satellite2 star9 star0
0
1
0 7 8 16
1
end_operator
begin_operator
turn_to satellite2 star9 star1
0
1
0 7 9 16
1
end_operator
begin_operator
turn_to satellite2 star9 star12
0
1
0 7 10 16
1
end_operator
begin_operator
turn_to satellite2 star9 star13
0
1
0 7 11 16
1
end_operator
begin_operator
turn_to satellite2 star9 star14
0
1
0 7 12 16
1
end_operator
begin_operator
turn_to satellite2 star9 star15
0
1
0 7 13 16
1
end_operator
begin_operator
turn_to satellite2 star9 star2
0
1
0 7 14 16
1
end_operator
begin_operator
turn_to satellite2 star9 star3
0
1
0 7 15 16
1
end_operator
0
