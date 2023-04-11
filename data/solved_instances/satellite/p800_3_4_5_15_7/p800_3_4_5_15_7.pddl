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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph4 - mode
	spectrograph0 - mode
	infrared2 - mode
	image1 - mode
	image3 - mode
	Star6 - direction
	Star9 - direction
	Star14 - direction
	GroundStation13 - direction
	GroundStation8 - direction
	Star2 - direction
	GroundStation11 - direction
	Star1 - direction
	GroundStation0 - direction
	Star10 - direction
	Star4 - direction
	GroundStation7 - direction
	Star5 - direction
	GroundStation12 - direction
	GroundStation3 - direction
	Star15 - direction
	Star16 - direction
	Phenomenon17 - direction
	Planet18 - direction
	Phenomenon19 - direction
	Planet20 - direction
	Planet21 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 GroundStation8)
	(calibration_target instrument0 GroundStation13)
	(supports instrument1 infrared2)
	(calibration_target instrument1 Star2)
	(calibration_target instrument1 Star4)
	(calibration_target instrument1 Star10)
	(supports instrument2 infrared2)
	(supports instrument2 spectrograph0)
	(supports instrument2 image3)
	(calibration_target instrument2 GroundStation12)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 Star1)
	(calibration_target instrument3 GroundStation12)
	(calibration_target instrument3 GroundStation11)
	(calibration_target instrument3 GroundStation0)
	(calibration_target instrument3 Star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Planet20)
	(supports instrument4 image3)
	(supports instrument4 image1)
	(calibration_target instrument4 GroundStation0)
	(calibration_target instrument4 Star1)
	(calibration_target instrument4 Star10)
	(supports instrument5 image3)
	(calibration_target instrument5 GroundStation7)
	(calibration_target instrument5 GroundStation3)
	(calibration_target instrument5 Star4)
	(calibration_target instrument5 Star10)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star6)
	(supports instrument6 spectrograph4)
	(supports instrument6 image3)
	(calibration_target instrument6 GroundStation12)
	(calibration_target instrument6 Star5)
	(supports instrument7 image1)
	(supports instrument7 image3)
	(supports instrument7 spectrograph4)
	(calibration_target instrument7 GroundStation12)
	(supports instrument8 infrared2)
	(supports instrument8 image3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 GroundStation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star6)
)
(:goal (and
	(pointing satellite0 GroundStation7)
	(pointing satellite2 GroundStation13)
	(have_image Star15 spectrograph0)
	(have_image Star16 image3)
	(have_image Phenomenon17 infrared2)
	(have_image Planet18 image1)
	(have_image Phenomenon19 image1)
	(have_image Planet20 image1)
	(have_image Planet21 infrared2)
))

)
