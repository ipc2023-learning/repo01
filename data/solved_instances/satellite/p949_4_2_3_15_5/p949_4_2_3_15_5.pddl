(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	thermograph2 - mode
	thermograph0 - mode
	image1 - mode
	GroundStation8 - direction
	GroundStation11 - direction
	GroundStation12 - direction
	Star10 - direction
	GroundStation3 - direction
	Star9 - direction
	GroundStation14 - direction
	Star2 - direction
	GroundStation4 - direction
	GroundStation0 - direction
	Star5 - direction
	GroundStation1 - direction
	GroundStation13 - direction
	Star7 - direction
	Star6 - direction
	Planet15 - direction
	Planet16 - direction
	Phenomenon17 - direction
	Star18 - direction
	Star19 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star6)
	(calibration_target instrument0 Star10)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 GroundStation1)
	(calibration_target instrument1 GroundStation14)
	(calibration_target instrument1 GroundStation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star10)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 Star6)
	(calibration_target instrument2 GroundStation0)
	(calibration_target instrument2 GroundStation4)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 Star6)
	(calibration_target instrument3 Star9)
	(calibration_target instrument3 GroundStation4)
	(calibration_target instrument3 GroundStation1)
	(calibration_target instrument3 GroundStation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star2)
	(supports instrument4 thermograph2)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 GroundStation14)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 GroundStation1)
	(calibration_target instrument5 Star5)
	(calibration_target instrument5 GroundStation0)
	(calibration_target instrument5 GroundStation4)
	(calibration_target instrument5 Star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star7)
	(supports instrument6 image1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 Star7)
	(calibration_target instrument6 GroundStation13)
	(supports instrument7 image1)
	(calibration_target instrument7 Star6)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star2)
)
(:goal (and
	(pointing satellite2 Star7)
	(pointing satellite3 Star2)
	(have_image Planet15 thermograph2)
	(have_image Planet16 image1)
	(have_image Phenomenon17 thermograph2)
	(have_image Star18 thermograph0)
	(have_image Star19 thermograph0)
))

)
