package main

import (
	"hackathon/api"
	"hackathon/database"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	db := database.Setup()
	r := gin.Default()
	r.Use(cors.Default())

	api.Setup(r.Group("/api"), db)
	r.Run()
}
