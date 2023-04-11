(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	thermograph3 - mode
	image1 - mode
	thermograph2 - mode
	thermograph0 - mode
	Star2 - direction
	Star9 - direction
	GroundStation7 - direction
	GroundStation4 - direction
	GroundStation3 - direction
	Star0 - direction
	GroundStation8 - direction
	GroundStation5 - direction
	Star1 - direction
	Star6 - direction
	Star10 - direction
	Star11 - direction
	Phenomenon12 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph3)
	(calibration_target instrument0 GroundStation4)
	(calibration_target instrument0 GroundStation8)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 GroundStation4)
	(calibration_target instrument1 GroundStation7)
	(calibration_target instrument1 GroundStation5)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 Star0)
	(calibration_target instrument2 GroundStation8)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation4)
	(supports instrument3 thermograph3)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 Star6)
	(calibration_target instrument3 GroundStation5)
	(calibration_target instrument3 GroundStation3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 GroundStation3)
	(calibration_target instrument4 Star0)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 GroundStation8)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star6)
	(supports instrument6 thermograph0)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 GroundStation8)
	(calibration_target instrument6 Star6)
	(calibration_target instrument6 Star0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star0)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 Star1)
	(calibration_target instrument7 Star6)
	(calibration_target instrument7 GroundStation5)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 Star6)
	(calibration_target instrument8 Star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star11)
)
(:goal (and
	(pointing satellite3 GroundStation3)
	(have_image Star10 thermograph3)
	(have_image Star11 thermograph3)
	(have_image Phenomenon12 thermograph2)
))

)
