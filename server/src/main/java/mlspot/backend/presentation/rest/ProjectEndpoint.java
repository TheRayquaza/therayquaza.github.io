package mlspot.backend.presentation.rest;

import io.vertx.ext.web.RoutingContext;
import mlspot.backend.converter.EntityResponseConverter;
import mlspot.backend.domain.entity.ProjectEntity;
import mlspot.backend.domain.service.ProjectService;
import mlspot.backend.errors.BadRequestError;
import mlspot.backend.errors.NotFoundError;
import mlspot.backend.errors.UnauthorizedError;
import mlspot.backend.exceptions.ProjectNotFoundException;
import mlspot.backend.presentation.rest.request.CreateProjectRequest;
import mlspot.backend.presentation.rest.request.ModifyProjectRequest;
import mlspot.backend.presentation.rest.response.ProjectResponse;
import mlspot.backend.presentation.rest.response.SuccessResponse;
import mlspot.backend.security.SecurityFilter;
import org.eclipse.microprofile.openapi.annotations.parameters.RequestBody;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.ws.rs.*;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

import static mlspot.backend.converter.EntityResponseConverter.Of;

@Path("/projects")
@Produces(MediaType.APPLICATION_JSON)
public class ProjectEndpoint {

    private final Logger logger = LoggerFactory.getLogger(ProjectEndpoint.class);

    @Inject
    ProjectService projectService;

    @Inject
    SecurityFilter securityFilter;

    @GET
    @Path("/")
    public Response getAllGProjectsEndpoint() {
        logger.info("[GET] /projects");
        List<ProjectEntity> projectEntities = projectService.getAllProjects();
        List<ProjectResponse> projectResponses = new ArrayList<>();
        projectEntities.forEach(p -> projectResponses.add(Of(p)));
        return Response
                .status(200)
                .entity(projectResponses)
                .build();
    }

    @POST
    @Path("/")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response createProjectEndpoint(@RequestBody CreateProjectRequest createProjectRequest, @Context RoutingContext routingContext) {
        logger.info("[POST] /projects");
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (createProjectRequest == null || createProjectRequest.getName() == null)
            return Response.status(400).entity(new BadRequestError()).build();
        ProjectEntity projectEntity = projectService.createProject(createProjectRequest.getName());
        if (projectEntity == null)
            return Response.status(400).entity(new BadRequestError()).build();
        return Response.status(200).entity(Of(projectEntity)).build();
    }

    @GET
    @Path("/{projectId}")
    public Response getProjectEndpoint(@PathParam(value = "projectId") Long projectId) {
        logger.info("[GET] /projects/" + projectId);
        if (projectId == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            return Response.status(200).entity(EntityResponseConverter.Of(projectService.getProject(projectId))).build();
        } catch (ProjectNotFoundException ignored) {
            logger.info("[404] could not find project " + projectId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @DELETE
    @Path("/{projectId}")
    public Response deleteProjectEndpoint(@PathParam(value = "projectId") Long projectId, @Context RoutingContext routingContext) {
        logger.info("[DELETE] /projects/" + projectId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (projectId == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            if (projectService.deleteProject(projectId))
                return Response.status(200).entity(new SuccessResponse()).build();
            return Response.status(400).entity(new BadRequestError()).build();
        } catch (ProjectNotFoundException ignored) {
            logger.info("[404] could not find project " + projectId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

    @PUT
    @Path("/{projectId}")
    public Response modifyProjectEndpoint(@RequestBody ModifyProjectRequest modifyProjectRequest, @PathParam(value = "projectId") Long projectId, @Context RoutingContext routingContext) {
        logger.info("[PUT] /projects/" + projectId);
        if (!securityFilter.checkPermission(routingContext))
            return Response.status(401).entity(new UnauthorizedError()).build();
        if (projectId == null || modifyProjectRequest == null)
            return Response.status(400).entity(new BadRequestError()).build();
        try {
            ProjectEntity projectEntity = projectService.modifyProject(modifyProjectRequest, projectId);
            return Response.status(200).entity(EntityResponseConverter.Of(projectEntity)).build();
        } catch (ProjectNotFoundException ignored) {
            logger.info("[404] could not find project " + projectId);
            return Response.status(404).entity(new NotFoundError()).build();
        }
    }

}
