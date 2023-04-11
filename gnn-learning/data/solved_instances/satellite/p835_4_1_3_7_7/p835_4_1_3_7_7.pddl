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
	thermograph2 - mode
	thermograph0 - mode
	GroundStation0 - direction
	GroundStation1 - direction
	Star2 - direction
	GroundStation4 - direction
	Star5 - direction
	GroundStation3 - direction
	Star6 - direction
	Star7 - direction
	Planet8 - direction
	Phenomenon9 - direction
	Planet10 - direction
	Star11 - direction
	Phenomenon12 - direction
	Planet13 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 GroundStation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation4)
	(supports instrument1 image1)
	(calibration_target instrument1 Star5)
	(calibration_target instrument1 Star6)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Phenomenon12)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 Star5)
	(calibration_target instrument2 Star6)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star11)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 Star6)
	(calibration_target instrument3 GroundStation3)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star11)
)
(:goal (and
	(pointing satellite0 GroundStation3)
	(pointing satellite1 Planet13)
	(pointing satellite2 Star6)
	(have_image Star7 thermograph0)
	(have_image Planet8 thermograph2)
	(have_image Phenomenon9 image1)
	(have_image Planet10 thermograph0)
	(have_image Star11 image1)
	(have_image Phenomenon12 image1)
	(have_image Planet13 image1)
))

)
