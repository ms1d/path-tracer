#include "httplib.h"
#include "json.hpp"
#include "endpoints/health.hpp"
#include "endpoints/submit_render.hpp"
#include "middleware/middleware_base.hpp"
#include "middleware/check_json.hpp"

using nlohmann::json;

int main() {
	httplib::Server svr;
	middleware mw;

	// Register middleware here
	mw.add_middleware(check_json, [](const Request&, Response&) {});

	// Call endpoints
	svr.Get("/health", mw.wrap_endpoint("/health", health));
	svr.Post("/submit-render", mw.wrap_endpoint("/submit-render", submit_render));

	svr.listen("localhost", 8080);

	return 1;
}
