(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared2 - mode
	image1 - mode
	spectrograph0 - mode
	image3 - mode
	spectrograph4 - mode
	Star10 - direction
	GroundStation3 - direction
	Star2 - direction
	GroundStation7 - direction
	Star1 - direction
	Star9 - direction
	GroundStation8 - direction
	GroundStation11 - direction
	Star4 - direction
	Star5 - direction
	Star6 - direction
	GroundStation0 - direction
	Phenomenon12 - direction
	Star13 - direction
	Star14 - direction
	Planet15 - direction
	Phenomenon16 - direction
	Planet17 - direction
	Planet18 - direction
)
(:init
	(supports instrument0 spectrograph4)
	(supports instrument0 infrared2)
	(supports instrument0 image3)
	(calibration_target instrument0 Star9)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 Star9)
	(calibration_target instrument1 Star4)
	(calibration_target instrument1 GroundStation3)
	(calibration_target instrument1 Star6)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star4)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph4)
	(calibration_target instrument2 GroundStation11)
	(supports instrument3 infrared2)
	(calibration_target instrument3 GroundStation0)
	(calibration_target instrument3 GroundStation7)
	(calibration_target instrument3 GroundStation3)
	(calibration_target instrument3 Star5)
	(supports instrument4 spectrograph4)
	(supports instrument4 infrared2)
	(calibration_target instrument4 GroundStation11)
	(calibration_target instrument4 Star9)
	(calibration_target instrument4 Star2)
	(supports instrument5 image3)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared2)
	(calibration_target instrument5 Star4)
	(calibration_target instrument5 Star9)
	(calibration_target instrument5 GroundStation7)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Planet18)
	(supports instrument6 infrared2)
	(calibration_target instrument6 Star9)
	(calibration_target instrument6 Star1)
	(supports instrument7 infrared2)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 Star4)
	(calibration_target instrument7 GroundStation8)
	(supports instrument8 image3)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 Star4)
	(calibration_target instrument8 GroundStation11)
	(supports instrument9 spectrograph0)
	(supports instrument9 image3)
	(supports instrument9 spectrograph4)
	(calibration_target instrument9 GroundStation0)
	(calibration_target instrument9 Star6)
	(calibration_target instrument9 Star5)
	(calibration_target instrument9 Star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation0)
)
(:goal (and
	(pointing satellite2 GroundStation3)
	(have_image Phenomenon12 image3)
	(have_image Star13 image3)
	(have_image Star14 spectrograph4)
	(have_image Planet15 image3)
	(have_image Phenomenon16 spectrograph4)
	(have_image Planet17 image3)
	(have_image Planet18 spectrograph0)
))

)
