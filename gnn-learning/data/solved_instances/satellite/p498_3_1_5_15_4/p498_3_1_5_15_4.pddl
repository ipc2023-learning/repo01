(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	image1 - mode
	spectrograph0 - mode
	image3 - mode
	infrared2 - mode
	spectrograph4 - mode
	GroundStation0 - direction
	Star1 - direction
	Star2 - direction
	GroundStation3 - direction
	Star4 - direction
	GroundStation8 - direction
	Star9 - direction
	Star10 - direction
	GroundStation11 - direction
	GroundStation13 - direction
	Star14 - direction
	GroundStation12 - direction
	Star5 - direction
	GroundStation7 - direction
	Star6 - direction
	Star15 - direction
	Star16 - direction
	Phenomenon17 - direction
	Planet18 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 GroundStation7)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Phenomenon17)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(calibration_target instrument1 GroundStation12)
	(calibration_target instrument1 GroundStation7)
	(calibration_target instrument1 Star6)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation13)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph4)
	(calibration_target instrument2 Star6)
	(calibration_target instrument2 GroundStation7)
	(calibration_target instrument2 Star5)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star4)
)
(:goal (and
	(have_image Star15 spectrograph0)
	(have_image Star16 image3)
	(have_image Phenomenon17 infrared2)
	(have_image Planet18 image1)
))

)
