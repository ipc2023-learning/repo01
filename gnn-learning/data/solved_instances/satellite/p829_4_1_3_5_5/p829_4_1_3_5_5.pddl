(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	image1 - mode
	thermograph0 - mode
	thermograph2 - mode
	GroundStation4 - direction
	GroundStation3 - direction
	Star2 - direction
	GroundStation0 - direction
	GroundStation1 - direction
	Star5 - direction
	Planet6 - direction
	Phenomenon7 - direction
	Star8 - direction
	Planet9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 Star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation4)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 GroundStation0)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Planet6)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 GroundStation1)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Planet6)
)
(:goal (and
	(pointing satellite0 GroundStation4)
	(pointing satellite1 Phenomenon7)
	(pointing satellite2 Star8)
	(pointing satellite3 GroundStation3)
	(have_image Star5 image1)
	(have_image Planet6 image1)
	(have_image Phenomenon7 thermograph0)
	(have_image Star8 image1)
	(have_image Planet9 thermograph2)
))

)
