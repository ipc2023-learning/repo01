(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	thermograph2 - mode
	image1 - mode
	thermograph0 - mode
	thermograph3 - mode
	GroundStation3 - direction
	GroundStation7 - direction
	Star10 - direction
	Star9 - direction
	GroundStation4 - direction
	Star11 - direction
	GroundStation8 - direction
	Star1 - direction
	Star6 - direction
	Star0 - direction
	GroundStation5 - direction
	Star2 - direction
	Planet12 - direction
	Planet13 - direction
	Planet14 - direction
	Planet15 - direction
	Star16 - direction
	Star17 - direction
	Planet18 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 Star10)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star17)
	(supports instrument1 thermograph3)
	(supports instrument1 thermograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 Star9)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 GroundStation4)
	(supports instrument3 thermograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 GroundStation4)
	(calibration_target instrument3 Star9)
	(calibration_target instrument3 GroundStation5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Planet15)
	(supports instrument4 thermograph3)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 GroundStation8)
	(calibration_target instrument4 Star11)
	(calibration_target instrument4 GroundStation5)
	(calibration_target instrument4 Star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Planet14)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 Star1)
	(supports instrument6 image1)
	(calibration_target instrument6 Star6)
	(supports instrument7 image1)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 Star2)
	(calibration_target instrument7 GroundStation5)
	(calibration_target instrument7 Star0)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star0)
)
(:goal (and
	(pointing satellite0 Planet13)
	(pointing satellite2 Star1)
	(pointing satellite3 Planet18)
	(have_image Planet12 thermograph0)
	(have_image Planet13 thermograph3)
	(have_image Planet14 image1)
	(have_image Planet15 thermograph3)
	(have_image Star16 thermograph0)
	(have_image Star17 thermograph0)
	(have_image Planet18 image1)
))

)
