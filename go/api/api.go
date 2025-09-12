package api

import (
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/jmoiron/sqlx"
)

type Service struct {
	DB *sqlx.DB
}

func Setup(r *gin.RouterGroup, db *sqlx.DB) {
	s := Service{DB: db}
	r.POST("/people", s.NewPerson)
	r.GET("/people", s.GetPeople)
	r.GET("/people/:id", s.GetPersonByID)
	r.DELETE("/people", s.DeletePersonByID)
	r.GET("/people/:id/scan", s.GetScansByPerson)

	r.POST("/scans", s.NewScan)
}

func (s *Service) NewScan(c *gin.Context) {
	scan := Fingerprint{}
	if err := c.ShouldBindJSON(&scan); err != nil {
		c.AbortWithStatus(400)
		return
	}
	err := s.DB.Get(&scan, "INSERT INTO scan (person_id, finger) VALUES (?) RETURNING *", scan.PersonID, scan.Finger)
	if err != nil {
		c.AbortWithError(500, err)
		return
	}
}

func (s *Service) GetScansByPerson(c *gin.Context) {
	key := c.Param("id")
	id, err := strconv.ParseInt(key, 10, 64)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}

	scans := []Fingerprint{}
	err = s.DB.Get(&scans, "SELECT FROM scan WHERE id = ?", id)
	if err != nil {
		c.AbortWithError(500, err)
		return
	}
}

func (s *Service) NewPerson(c *gin.Context) {
	person := Person{}
	if err := c.ShouldBindJSON(&person); err != nil {
		c.AbortWithStatus(400)
		return
	}

	err := s.DB.Get(&person, "INSERT INTO person (name, email, phone, role, address, age) VALUES (?, ?, ?, ?, ?, ?) RETURNING *", person.Name, person.Email, person.Phone, person.Role, person.Address, person.Age)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}
	c.JSON(201, person)
}

func (s *Service) GetPeople(c *gin.Context) {
	people := []Person{}
	err := s.DB.Select(&people, "SELECT * FROM person")
	if err != nil {
		c.AbortWithError(400, err)
		return
	}

	c.JSON(200, people)
}

func (s *Service) GetPersonByID(c *gin.Context) {
	key := c.Param("id")
	id, err := strconv.ParseInt(key, 10, 64)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}
	person := Person{}
	err = s.DB.Get(&person, "SELECT * FROM person WHERE id = ?", id)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}

	c.JSON(200, person)
}

func (s *Service) DeletePersonByID(c *gin.Context) {
	key := c.Param("id")
	id, err := strconv.ParseInt(key, 10, 64)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}

	_, err = s.DB.Exec("DELETE FROM person where id = ?", id)
	if err != nil {
		c.AbortWithError(400, err)
		return
	}

	c.Status(204)
}
