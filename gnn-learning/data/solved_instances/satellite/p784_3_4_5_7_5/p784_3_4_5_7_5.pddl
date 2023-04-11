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
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	image1 - mode
	image3 - mode
	spectrograph4 - mode
	spectrograph0 - mode
	infrared2 - mode
	GroundStation3 - direction
	Star2 - direction
	Star5 - direction
	Star4 - direction
	Star1 - direction
	GroundStation0 - direction
	Star6 - direction
	Planet7 - direction
	Phenomenon8 - direction
	Star9 - direction
	Phenomenon10 - direction
	Star11 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image3)
	(calibration_target instrument0 GroundStation0)
	(calibration_target instrument0 Star6)
	(supports instrument1 image3)
	(supports instrument1 spectrograph4)
	(supports instrument1 image1)
	(calibration_target instrument1 Star2)
	(calibration_target instrument1 Star6)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 GroundStation0)
	(calibration_target instrument2 Star6)
	(supports instrument3 spectrograph4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 GroundStation0)
	(calibration_target instrument3 Star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star5)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 Star1)
	(calibration_target instrument4 GroundStation0)
	(supports instrument5 spectrograph4)
	(calibration_target instrument5 Star4)
	(calibration_target instrument5 Star5)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(supports instrument6 infrared2)
	(calibration_target instrument6 Star1)
	(supports instrument7 spectrograph4)
	(supports instrument7 image3)
	(supports instrument7 image1)
	(calibration_target instrument7 Star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation3)
	(supports instrument8 spectrograph4)
	(supports instrument8 image1)
	(supports instrument8 infrared2)
	(calibration_target instrument8 Star6)
	(calibration_target instrument8 GroundStation0)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Planet7)
)
(:goal (and
	(pointing satellite0 GroundStation3)
	(have_image Planet7 image3)
	(have_image Phenomenon8 spectrograph4)
	(have_image Star9 spectrograph4)
	(have_image Phenomenon10 spectrograph4)
	(have_image Star11 image1)
))

)
