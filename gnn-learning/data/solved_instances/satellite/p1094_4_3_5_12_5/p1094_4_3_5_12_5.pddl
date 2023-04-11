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
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	thermograph3 - mode
	thermograph2 - mode
	image1 - mode
	image4 - mode
	thermograph0 - mode
	GroundStation9 - direction
	GroundStation2 - direction
	Star4 - direction
	Star8 - direction
	GroundStation1 - direction
	Star5 - direction
	Star10 - direction
	Star6 - direction
	GroundStation3 - direction
	Star7 - direction
	Star11 - direction
	Star0 - direction
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
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star6)
	(supports instrument3 image4)
	(supports instrument3 image1)
	(calibration_target instrument3 Star7)
	(calibration_target instrument3 Star8)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 Star11)
	(supports instrument5 thermograph2)
	(supports instrument5 image4)
	(calibration_target instrument5 Star8)
	(calibration_target instrument5 Star7)
	(calibration_target instrument5 GroundStation3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star6)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 GroundStation1)
	(calibration_target instrument6 Star7)
	(calibration_target instrument6 GroundStation3)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 Star5)
	(supports instrument8 image4)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 Star11)
	(calibration_target instrument8 Star6)
	(calibration_target instrument8 Star10)
	(calibration_target instrument8 Star7)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star7)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 Star7)
	(calibration_target instrument9 GroundStation3)
	(supports instrument10 thermograph3)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 Star0)
	(calibration_target instrument10 Star11)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star5)
)
(:goal (and
	(pointing satellite2 Phenomenon16)
	(pointing satellite3 Star10)
	(have_image Star12 image4)
	(have_image Phenomenon13 image1)
	(have_image Phenomenon14 image4)
	(have_image Phenomenon15 image4)
	(have_image Phenomenon16 thermograph3)
))

)
