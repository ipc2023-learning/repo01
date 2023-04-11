(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	thermograph0 - mode
	image4 - mode
	thermograph3 - mode
	thermograph2 - mode
	Star7 - direction
	Star11 - direction
	Star6 - direction
	GroundStation1 - direction
	GroundStation9 - direction
	Star0 - direction
	GroundStation2 - direction
	Star5 - direction
	Star4 - direction
	Star8 - direction
	GroundStation3 - direction
	Star10 - direction
	Star12 - direction
	Phenomenon13 - direction
	Phenomenon14 - direction
	Phenomenon15 - direction
	Phenomenon16 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image4)
	(calibration_target instrument0 Star6)
	(calibration_target instrument0 Star0)
	(supports instrument1 image4)
	(calibration_target instrument1 GroundStation2)
	(calibration_target instrument1 Star4)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 Star5)
	(calibration_target instrument2 Star4)
	(calibration_target instrument2 GroundStation3)
	(calibration_target instrument2 GroundStation1)
	(supports instrument3 image4)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 Star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star6)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 Star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Phenomenon13)
	(supports instrument5 image4)
	(supports instrument5 thermograph2)
	(supports instrument5 image1)
	(calibration_target instrument5 GroundStation9)
	(calibration_target instrument5 Star5)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 Star5)
	(calibration_target instrument6 Star8)
	(calibration_target instrument6 GroundStation2)
	(calibration_target instrument6 Star0)
	(supports instrument7 thermograph2)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 Star10)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 Star5)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star5)
	(supports instrument9 image1)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 Star8)
	(supports instrument10 thermograph0)
	(supports instrument10 thermograph3)
	(supports instrument10 image1)
	(calibration_target instrument10 GroundStation3)
	(supports instrument11 image1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 GroundStation3)
	(calibration_target instrument11 Star8)
	(calibration_target instrument11 Star4)
	(supports instrument12 thermograph2)
	(supports instrument12 thermograph3)
	(supports instrument12 image4)
	(calibration_target instrument12 Star10)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Phenomenon14)
)
(:goal (and
	(have_image Star12 image4)
	(have_image Phenomenon13 image1)
	(have_image Phenomenon14 image4)
	(have_image Phenomenon15 image4)
	(have_image Phenomenon16 thermograph3)
))

)
